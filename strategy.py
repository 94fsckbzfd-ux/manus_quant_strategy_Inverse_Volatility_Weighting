# -*- coding: utf-8 -*-
"""
量化策略 v2.1 - 逆波动率加权 + RSRS 择时
升级日志：
  - [P0] 修复止损/止盈成本基准计算错误，引入 cost_prices 记录真实买入均价
  - [P0] 修复数据异常被完全吞掉问题，改为分类捕获并推送告警
  - [P1] 冷静期改为绝对结束日期，解决节假日场景计数偏差
  - [P1] 加入交易成本（印花税 0.1% + 佣金万3 + 滑点 0.1%）
  - [P2] 进攻选股动量窗口从 11 日改为 20 日，减少追涨杀跌
  - [P2] 增加 AKShare 备用数据源，主备自动切换
  - [v2.1] 参数网格优化：RSRS 阈值 0.8→0.6，动量窗口 20→11，波动率窗口保持 21
  - [v2.1] 新增进攻信号假触发率统计（实盘观察项：11日动量在震荡期的表现）
"""
import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ================= 策略配置 =================
STOCK_NAMES = {
    '512480': '半导体ETF',
    '516820': '氢能ETF',       # 替换 159206（中证1000），降低进攻池内部相关性
    '159755': '电池ETF',
    '159819': '人工智能ETF',
    '512890': '红利低波ETF',
    '518880': '黄金ETF',
    '159928': '中证消费ETF',  # 替换 513100（纳指100），提升防御池真实防御属性
    '511880': '银华日利',
    '510300': '沪深300ETF'
}

ALPHA_POOL = ['512480', '516820', '159755', '159819']  # 中证1000 → 氢能ETF
BETA_POOL  = ['512890', '518880', '159928']             # 纳指100 → 中证消费ETF
MARKET_ANCHOR = '510300'
CASH_CODE     = '511880'

N_DAYS, M_DAYS = 18, 250
S_BUY          = 0.6   # [v2.1] 网格优化：0.8 → 0.6（夏普最优）
MOMENTUM_DAYS  = 11   # [v2.1] 网格优化：20 → 11（夏普最优）
                       # ⚠️ [实盘观察项] 11日动量窗口在震荡行情中假触发率较高。
                       #    若未来半年出现区间震荡，请记录每次进攻信号触发后
                       #    实际持仓的盈亏结果，评估是否需要回调至 20 日。
MAX_DRAWDOWN_LIMIT = 0.15    # 组合整体止损线
INITIAL_CASH   = 100000.0
STATE_FILE     = 'portfolio_state.json'
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN', '')

# 进攻仓位双边约束
ALPHA_MAX = 0.60   # 进攻仓位上限：避免单仓押注
ALPHA_MIN = 0.30   # 进攻仓位下限：RSRS已确认强势行情，进攻仓过小失去意义

# 单标的差异化止盈止损（进攻 vs 防御）
ALPHA_STOP_LOSS    = -0.12
ALPHA_TAKE_PROFIT  =  0.25
BETA_STOP_LOSS     = -0.06
BETA_TAKE_PROFIT   =  0.15

# [P1] 交易成本
COMMISSION_RATE = 0.0003   # 佣金：万3（双向）
STAMP_DUTY_RATE = 0.001    # 印花税：0.1%（仅卖出）
SLIPPAGE_RATE   = 0.001    # 滑点估算：0.1%

# ================= 数据获取 =================
def get_price_tencent(code, count):
    """腾讯财经数据源（主）"""
    market = 'sh' if str(code).startswith(('5', '6', '11', '588')) else 'sz'
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,{count},qfq"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        data_node = data.get("data", {}).get(f"{market}{code}", {})
        k_list = data_node.get("qfqday", data_node.get("day", []))
        if not k_list:
            print(f"[WARNING] 腾讯数据为空：{code}")
            return pd.DataFrame()
        df = pd.DataFrame(k_list, columns=['date','open','close','high','low','vol','amt','turnover'][:len(k_list[0])])
        for col in ['open','close','high','low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['close'])
    except requests.exceptions.Timeout:
        print(f"[ERROR] 腾讯数据获取超时：{code}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] 腾讯数据获取失败：{code}，原因：{e}")
        return pd.DataFrame()

def get_price_akshare(code, count):
    """AKShare 备用数据源"""
    try:
        import akshare as ak
        df_raw = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
        if df_raw is None or df_raw.empty:
            print(f"[WARNING] AKShare 数据为空：{code}")
            return pd.DataFrame()
        df = df_raw.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'vol'})
        df = df[['date','open','close','high','low','vol']].tail(count).reset_index(drop=True)
        for col in ['open','close','high','low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['close'])
    except Exception as e:
        print(f"[ERROR] AKShare 数据获取失败：{code}，原因：{e}")
        return pd.DataFrame()

def get_price(code, count):
    """[P2] 主备数据源自动切换"""
    df = get_price_tencent(code, count)
    if df.empty:
        print(f"[WARN] 腾讯数据源失败，切换备用源：{code}")
        df = get_price_akshare(code, count)
    return df

# ================= 工具函数 =================
def get_stock_display(code):
    name = STOCK_NAMES.get(str(code), "未知标的")
    return f"{code}（{name}）"

def send_pushplus(title, content):
    if not PUSHPLUS_TOKEN: return
    url  = 'http://www.pushplus.plus/send'
    data = {"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"}
    try:
        requests.post(url, json=data, timeout=10)
    except Exception as e:
        print(f"[ERROR] PushPlus 推送失败：{e}")

def get_rsrs_signal(error_logs):
    """[P0] 数据不足时返回 None 并记录告警，而非静默返回 0"""
    prices = get_price(MARKET_ANCHOR, count=N_DAYS + M_DAYS)
    if prices.empty or len(prices) < N_DAYS + M_DAYS:
        msg = f"[ERROR] RSRS 基准数据不足（获取 {len(prices)} 条，需要 {N_DAYS + M_DAYS} 条），本次运行中止。"
        print(msg)
        error_logs.append(msg)
        return None
    betas = []
    for i in range(len(prices) - N_DAYS + 1):
        df = prices.iloc[i:i+N_DAYS]
        model = LinearRegression().fit(df['low'].values.reshape(-1, 1), df['high'].values)
        betas.append(model.coef_[0])
    return (betas[-1] - np.mean(betas)) / np.std(betas) if np.std(betas) != 0 else 0

def build_position_block(state, current_prices, portfolio_value):
    """生成当前持仓的 HTML 展示块"""
    positions = state.get('positions', {})
    cost_prices = state.get('cost_prices', {})
    lines = ["<h3>📋 当前持仓：</h3>"]
    if not positions:
        lines.append("<p>🈳 当前空仓（持有现金）</p>")
    else:
        lines.append("<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;'>")
        lines.append("<tr><th>代码</th><th>名称</th><th>持仓份额</th><th>成本价</th><th>现价</th><th>浮盈亏</th><th>市值</th><th>占比</th></tr>")
        for code, shares in positions.items():
            price      = current_prices.get(code, 0)
            cost_price = cost_prices.get(code, 0)
            market_val = shares * price
            ratio      = market_val / portfolio_value if portfolio_value > 0 else 0
            pnl_ratio  = (price - cost_price) / cost_price if cost_price > 0 else 0
            pnl_str    = f"{pnl_ratio:+.2%}" if cost_price > 0 else "N/A"
            name       = STOCK_NAMES.get(str(code), "未知标的")
            lines.append(
                f"<tr><td>{code}</td><td>{name}</td><td>{shares:,}</td>"
                f"<td>¥{cost_price:.4f}</td><td>¥{price:.4f}</td><td>{pnl_str}</td>"
                f"<td>¥{market_val:,.2f}</td><td>{ratio:.2%}</td></tr>"
            )
        lines.append("</table>")
    cash       = state.get('cash', 0)
    cash_ratio = cash / portfolio_value if portfolio_value > 0 else 0
    lines.append(f"<p>💰 可用现金: ¥{cash:,.2f}（占比 {cash_ratio:.2%}）</p>")
    return "".join(lines)

def check_position_alerts(state, current_prices, logs):
    """
    [P0] 使用真实买入均价（cost_prices）计算浮盈亏，
    并按进攻/防御引擎应用差异化止盈止损阈值。
    """
    positions   = state.get('positions', {})
    cost_prices = state.get('cost_prices', {})
    alert_triggered = False

    for code, shares in positions.items():
        current_price = current_prices.get(code, 0)
        cost_price    = cost_prices.get(code, 0)
        if current_price <= 0 or cost_price <= 0 or shares <= 0:
            continue

        pnl_ratio = (current_price - cost_price) / cost_price
        name      = STOCK_NAMES.get(str(code), "未知标的")

        if code in ALPHA_POOL:
            stop_loss_limit    = ALPHA_STOP_LOSS
            take_profit_limit  = ALPHA_TAKE_PROFIT
            engine_label       = "进攻"
        elif code in BETA_POOL:
            stop_loss_limit    = BETA_STOP_LOSS
            take_profit_limit  = BETA_TAKE_PROFIT
            engine_label       = "防御"
        else:
            stop_loss_limit    = -0.15
            take_profit_limit  =  0.20
            engine_label       = "默认"

        if pnl_ratio <= stop_loss_limit:
            logs.append(
                f"🚨 <b>止损警报！[{engine_label}] {code}（{name}）</b> "
                f"成本 ¥{cost_price:.4f} → 现价 ¥{current_price:.4f}，"
                f"浮亏 <b>{pnl_ratio:.2%}</b>，触及 {stop_loss_limit:.0%} 止损线，建议立即减仓或清仓！"
            )
            alert_triggered = True
        elif pnl_ratio >= take_profit_limit:
            logs.append(
                f"🎯 <b>止盈提醒！[{engine_label}] {code}（{name}）</b> "
                f"成本 ¥{cost_price:.4f} → 现价 ¥{current_price:.4f}，"
                f"浮盈 <b>{pnl_ratio:.2%}</b>，触及 +{take_profit_limit:.0%} 止盈线，建议考虑分批止盈。"
            )
            alert_triggered = True

    return alert_triggered

# ================= 核心逻辑 =================
def market_trade():
    weekday   = datetime.now().weekday()
    is_friday = (weekday == 4)
    today_str = datetime.now().strftime('%Y-%m-%d')

    # 读取或初始化状态
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        state = {
            "cash": INITIAL_CASH,
            "positions": {},
            "cost_prices": {},       # [P0] 记录每只标的的真实买入均价
            "peak_value": INITIAL_CASH,
            "is_cooling": False,
            "cool_end_date": ""      # [P1] 冷静期绝对结束日期
        }

    # 兼容旧版 state（无 cost_prices / cool_end_date 字段）
    if 'cost_prices' not in state:
        state['cost_prices'] = {}
    if 'cool_end_date' not in state:
        state['cool_end_date'] = ""
    if 'cool_down_weeks' in state:
        del state['cool_down_weeks']

    # 获取所有标的最新价格
    error_logs = []
    current_prices = {}
    all_codes = set(ALPHA_POOL + BETA_POOL + [CASH_CODE] + list(state['positions'].keys()))
    for code in all_codes:
        df = get_price(code, 5)
        if not df.empty:
            current_prices[code] = df['close'].iloc[-1]
        else:
            error_logs.append(f"[ERROR] 无法获取 {get_stock_display(code)} 的价格数据（主备源均失败）。")
            current_prices[code] = 0

    portfolio_value = state['cash'] + sum(
        state['positions'].get(c, 0) * current_prices.get(c, 0) for c in state['positions']
    )
    if portfolio_value > state['peak_value']:
        state['peak_value'] = portfolio_value
    current_drawdown = (portfolio_value - state['peak_value']) / state['peak_value'] if state['peak_value'] > 0 else 0

    logs = [
        f"📅 运行日期: {today_str} (周{weekday+1})",
        f"📊 账户总净值: ¥{portfolio_value:,.2f} | 峰值: ¥{state['peak_value']:,.2f}",
        f"📉 当前回撤: {current_drawdown:+.2%}",
    ]

    # 数据获取错误告警
    if error_logs:
        logs.append("<p style='color:red;'><b>⚠️ 数据获取警告：</b></p><ul>")
        for e in error_logs:
            logs.append(f"<li>{e}</li>")
        logs.append("</ul>")

    # 插入当前持仓展示（含成本价和浮盈亏）
    logs.append(build_position_block(state, current_prices, portfolio_value))

    # --- 单标的止盈止损检查（每日执行）---
    alert_triggered = check_position_alerts(state, current_prices, logs)
    if alert_triggered:
        send_pushplus("⚠️ 持仓止盈止损提醒", "<br>".join(logs))

    # --- A. 组合整体紧急风控（每日检查）---
    if current_drawdown <= -MAX_DRAWDOWN_LIMIT:
        logs.append("<b>🚨 紧急警报：组合整体回撤触及止损线！立即清空所有持仓。</b>")
        # [P1] 冷静期改为绝对结束日期
        cool_end = (datetime.now() + timedelta(weeks=2)).strftime('%Y-%m-%d')
        state['is_cooling']    = True
        state['cool_end_date'] = cool_end
        for stock in list(state['positions'].keys()):
            sell_val  = state['positions'][stock] * current_prices.get(stock, 0)
            sell_cost = sell_val * (COMMISSION_RATE + STAMP_DUTY_RATE + SLIPPAGE_RATE)
            state['cash'] += sell_val - sell_cost
        state['positions']  = {}
        state['cost_prices'] = {}
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        send_pushplus("🚨 紧急风控触发：全仓清空！", "<br>".join(logs))
        return

    # --- B. 冷静期维护（[P1] 基于绝对日期判断）---
    if state['is_cooling']:
        cool_end = state.get('cool_end_date', '')
        if today_str >= cool_end:
            state['is_cooling'] = False
            logs.append(f"✅ 冷静期结束（{cool_end}），恢复交易！")
        else:
            logs.append(f"❄️ 冷静期中，结束日期：{cool_end}，交易暂停。")
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
            send_pushplus("日间监控 - 冷静期中", "<br>".join(logs))
            return

    # --- C. RSRS 信号计算（[P0] 数据不足时中止）---
    rsrs_signal = get_rsrs_signal(error_logs)
    if rsrs_signal is None:
        logs.append("<b>⚠️ RSRS 数据不足，本次策略运行中止，请人工检查数据源。</b>")
        send_pushplus("⚠️ 策略运行中止 - 数据不足", "<br>".join(logs))
        return
    logs.append(f"📡 RSRS信号: {rsrs_signal:.4f} (阈值: {S_BUY})")

    if not is_friday:
        send_pushplus("日间监控报告 - 趋势观察", "<br>".join(logs))
        return

    # ===== 周五正式调仓逻辑 =====
    selected_stocks = []

    # 进攻引擎：[v2.1] 动量窗口 11 日（网格最优）
    # ⚠️ [实盘观察项] 统计进攻信号触发频率，用于评估震荡期假触发率
    signal_log = state.setdefault('signal_log', [])
    signal_entry = {
        'date': today_str,
        'rsrs': round(rsrs_signal, 4),
        'triggered': rsrs_signal > S_BUY
    }
    signal_log.append(signal_entry)
    # 只保留最近 26 周（半年）的记录
    state['signal_log'] = signal_log[-26:]

    # 统计近 26 周假触发率（触发进攻但随后亏损视为假触发，此处仅统计触发频率）
    recent_triggers = [e for e in state['signal_log'] if e['triggered']]
    trigger_rate    = len(recent_triggers) / len(state['signal_log']) if state['signal_log'] else 0
    logs.append(
        f"📊 <b>[观察项] 近期进攻信号触发率：{trigger_rate:.1%}</b>"
        f"（近 {len(state['signal_log'])} 周中触发 {len(recent_triggers)} 次）"
    )

    if rsrs_signal > S_BUY:
        scores = {}
        for stock in ALPHA_POOL:
            p = get_price(stock, count=MOMENTUM_DAYS + 1)
            if len(p) >= MOMENTUM_DAYS + 1:
                ret = p['close'].iloc[-1] / p['close'].iloc[0] - 1
                if ret > 0.03:
                    scores[stock] = ret
        if scores:
            best_alpha = sorted(scores, key=scores.get, reverse=True)[0]
            selected_stocks.append(best_alpha)
            logs.append(f"⚔️ 进攻引擎: 选中 {get_stock_display(best_alpha)}")

    # 防御引擎
    beta_scores = {}
    for stock in BETA_POOL:
        p = get_price(stock, count=21)
        if len(p) >= 21:
            ret = p['close'].iloc[-1] / p['close'].iloc[0] - 1
            vol = np.std(p['close'].pct_change().dropna())
            beta_scores[stock] = ret / vol if vol > 0 else 0
    if beta_scores:
        best_beta = sorted(beta_scores, key=beta_scores.get, reverse=True)[0]
        selected_stocks.append(best_beta)
        logs.append(f"🛡️ 防御引擎: 选中 {get_stock_display(best_beta)}")

    if not selected_stocks:
        selected_stocks = [CASH_CODE]
        logs.append(f"💤 避险至 {get_stock_display(CASH_CODE)}")

    # 波动率加权
    weights, vols = {}, {}
    for stock in selected_stocks:
        p   = get_price(stock, count=21)
        vol = np.std(p['close'].pct_change().dropna()) if not p.empty else 0
        vols[stock] = vol if vol > 0 else 0.02
    inv_vol_sum = sum(1.0 / v for v in vols.values())
    for stock in selected_stocks:
        weights[stock] = (1.0 / vols[stock]) / inv_vol_sum

    # 进攻仓位双边约束 + 归一化
    if len(selected_stocks) == 2:
        alpha_stocks = [s for s in selected_stocks if s in ALPHA_POOL]
        beta_stocks  = [s for s in selected_stocks if s in BETA_POOL]
        if alpha_stocks and beta_stocks:
            a_code = alpha_stocks[0]
            b_code = beta_stocks[0]
            raw_alpha_w = weights[a_code]
            clipped_alpha_w = max(ALPHA_MIN, min(ALPHA_MAX, raw_alpha_w))
            weights[a_code] = clipped_alpha_w
            weights[b_code] = 1.0 - clipped_alpha_w
            if abs(clipped_alpha_w - raw_alpha_w) > 0.001:
                logs.append(
                    f"⚖️ 仓位约束触发：{get_stock_display(a_code)} 进攻仓原始权重 {raw_alpha_w:.2%} "
                    f"→ 裁剪至 {clipped_alpha_w:.2%}（区间 [{ALPHA_MIN:.0%}, {ALPHA_MAX:.0%}]）"
                )

    # 调仓建议
    logs.append("<h3>🛒 本周正式调仓建议：</h3><ul>")
    for stock, weight in weights.items():
        logs.append(f"<li>标的: <b>{get_stock_display(stock)}</b> | 建议配比: <b>{weight:.2%}</b></li>")
    logs.append("</ul>")

    # [P1] 虚拟盘调仓：计入交易成本
    # 先卖出不在新组合中的标的
    for stock in list(state['positions'].keys()):
        if stock not in weights:
            sell_val  = state['positions'][stock] * current_prices.get(stock, 0)
            sell_cost = sell_val * (COMMISSION_RATE + STAMP_DUTY_RATE + SLIPPAGE_RATE)
            state['cash'] += sell_val - sell_cost
            del state['positions'][stock]
            if stock in state['cost_prices']:
                del state['cost_prices'][stock]

    # 再买入新组合标的
    for stock, weight in weights.items():
        curr_price = current_prices.get(stock, 0)
        if curr_price <= 0:
            continue
        if stock == CASH_CODE:
            shares = int(portfolio_value * weight / curr_price)
        else:
            shares = int(portfolio_value * weight / curr_price / 100) * 100
        if shares < 100 and stock != CASH_CODE:
            continue

        old_shares = state['positions'].get(stock, 0)
        # 卖出旧仓
        if old_shares > 0:
            sell_val  = old_shares * curr_price
            sell_cost = sell_val * (COMMISSION_RATE + STAMP_DUTY_RATE + SLIPPAGE_RATE)
            state['cash'] += sell_val - sell_cost
        # 买入新仓
        buy_val  = shares * curr_price
        buy_cost = buy_val * (COMMISSION_RATE + SLIPPAGE_RATE)
        state['cash']               -= buy_val + buy_cost
        state['positions'][stock]    = shares
        state['cost_prices'][stock]  = curr_price   # [P0] 记录真实买入均价

    # 调仓后更新持仓展示
    new_portfolio_value = state['cash'] + sum(
        state['positions'].get(c, 0) * current_prices.get(c, 0) for c in state['positions']
    )
    logs.append("<h3>📋 调仓后最新持仓：</h3>")
    logs.append(build_position_block(state, current_prices, new_portfolio_value))

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
    send_pushplus(f"🚀 周五正式调仓报告 ({today_str})", "<br>".join(logs))

if __name__ == "__main__":
    market_trade()

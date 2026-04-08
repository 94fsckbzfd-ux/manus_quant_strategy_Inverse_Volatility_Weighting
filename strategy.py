# -*- coding: utf-8 -*-
import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ================= 策略配置 =================
# 股票代码与名称映射
STOCK_NAMES = {
    '512480': '半导体ETF',
    '159206': '中证1000ETF',
    '159755': '电池ETF',
    '516690': '氢能ETF',
    '159819': '人工智能ETF',
    '512890': '红利低波ETF',
    '518880': '黄金ETF',
    '513100': '纳指100ETF',
    '511880': '银华日利',
    '510300': '沪深300ETF'
}

ALPHA_POOL = ['512480', '159206', '159755', '159819']
BETA_POOL = ['512890', '518880', '513100']
MARKET_ANCHOR = '510300'
CASH_CODE = '511880'

N_DAYS, M_DAYS = 18, 250
S_BUY = 0.8
MAX_DRAWDOWN_LIMIT = 0.15
INITIAL_CASH = 100000.0
STATE_FILE = 'portfolio_state.json'
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN', '')

# ================= 工具函数 =================
def get_stock_display(code):
    name = STOCK_NAMES.get(str(code), "未知标的")
    return f"{code} ({name})"

def get_price_tencent(code, count):
    market = 'sh' if str(code).startswith(('5', '6', '11', '588')) else 'sz'
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,{count},qfq"
    try:
        resp = requests.get(url, timeout=10).json()
        data_node = resp.get("data", {}).get(f"{market}{code}", {})
        k_list = data_node.get("qfqday", data_node.get("day", []))
        if not k_list: return pd.DataFrame()
        df = pd.DataFrame(k_list, columns=['date', 'open', 'close', 'high', 'low', 'vol', 'amt', 'turnover'][:len(k_list[0])])
        for col in ['open', 'close', 'high', 'low']: df[col] = df[col].astype(float)
        return df
    except: return pd.DataFrame()

def send_pushplus(title, content):
    if not PUSHPLUS_TOKEN: return
    url = 'http://www.pushplus.plus/send'
    data = {"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"}
    requests.post(url, json=data)

def get_rsrs_signal():
    prices = get_price_tencent(MARKET_ANCHOR, count=N_DAYS + M_DAYS)
    if len(prices) < N_DAYS + M_DAYS: return 0
    betas = []
    for i in range(len(prices) - N_DAYS + 1):
        df = prices.iloc[i:i+N_DAYS]
        model = LinearRegression().fit(df['low'].values.reshape(-1, 1), df['high'].values)
        betas.append(model.coef_[0])
    return (betas[-1] - np.mean(betas)) / np.std(betas) if np.std(betas) != 0 else 0

def build_position_block(state, current_prices, portfolio_value):
    """生成当前持仓的 HTML 展示块，用于插入每次推送通知。"""
    positions = state.get('positions', {})
    lines = ["<h3>📋 当前持仓：</h3>"]
    if not positions:
        lines.append("<p>🈳 当前空仓（持有现金）</p>")
    else:
        lines.append("<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;'>")
        lines.append("<tr><th>代码</th><th>名称</th><th>持仓份额</th><th>现价</th><th>市值</th><th>占比</th></tr>")
        for code, shares in positions.items():
            price = current_prices.get(code, 0)
            market_val = shares * price
            ratio = market_val / portfolio_value if portfolio_value > 0 else 0
            name = STOCK_NAMES.get(str(code), "未知标的")
            lines.append(
                f"<tr><td>{code}</td><td>{name}</td><td>{shares:,}</td>"
                f"<td>¥{price:.4f}</td><td>¥{market_val:,.2f}</td><td>{ratio:.2%}</td></tr>"
            )
        lines.append("</table>")
    cash = state.get('cash', 0)
    cash_ratio = cash / portfolio_value if portfolio_value > 0 else 0
    lines.append(f"<p>💰 可用现金: ¥{cash:,.2f}（占比 {cash_ratio:.2%}）</p>")
    return "".join(lines)

# ================= 核心逻辑 =================
def market_trade():
    weekday = datetime.now().weekday()  # 0-6, 4是周五
    is_friday = (weekday == 4)

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else:
        state = {"cash": INITIAL_CASH, "positions": {}, "peak_value": INITIAL_CASH, "is_cooling": False, "cool_down_weeks": 0}

    current_prices = {}
    all_codes = set(ALPHA_POOL + BETA_POOL + [CASH_CODE] + list(state['positions'].keys()))
    for code in all_codes:
        df = get_price_tencent(code, 1)
        current_prices[code] = df['close'].iloc[-1] if not df.empty else 0

    portfolio_value = state['cash'] + sum(state['positions'].get(c, 0) * current_prices.get(c, 0) for c in state['positions'])
    if portfolio_value > state['peak_value']: state['peak_value'] = portfolio_value
    current_drawdown = (portfolio_value - state['peak_value']) / state['peak_value'] if state['peak_value'] > 0 else 0

    logs = [f"📅 运行日期: {datetime.now().strftime('%Y-%m-%d')} (周{weekday+1})"]
    logs.append(f"📊 账户总净值: ¥{portfolio_value:,.2f} | 峰值: ¥{state['peak_value']:,.2f}")
    logs.append(f"📉 当前回撤: {current_drawdown:+.2%}")

    # 插入当前持仓展示
    logs.append(build_position_block(state, current_prices, portfolio_value))

    # --- A. 紧急风控 (每日检查) ---
    if current_drawdown <= -MAX_DRAWDOWN_LIMIT:
        logs.append("<b>🚨 紧急警报：回撤触及止损线！立即清空所有持仓。</b>")
        state['is_cooling'], state['cool_down_weeks'] = True, 2
        for stock in list(state['positions'].keys()):
            state['cash'] += state['positions'][stock] * current_prices[stock]
        state['positions'] = {}
        with open(STATE_FILE, 'w') as f: json.dump(state, f)
        send_pushplus("🚨 紧急风控触发：全仓清空！", "<br>".join(logs))
        return

    # --- B. 冷静期维护 ---
    if state['is_cooling']:
        if is_friday: state['cool_down_weeks'] -= 1
        if state['cool_down_weeks'] <= 0:
            state['is_cooling'] = False
            logs.append("❄️ 冷静期结束，系统重启。")
        else:
            logs.append(f"❄️ 冷静期中，剩余 {state['cool_down_weeks']} 周。")
            with open(STATE_FILE, 'w') as f: json.dump(state, f)
            send_pushplus("日间监控 - 冷静期中", "<br>".join(logs))
            return

    # --- C. 策略逻辑 (仅周五生成建议) ---
    rsrs_signal = get_rsrs_signal()
    logs.append(f"📡 RSRS信号: {rsrs_signal:.4f} (阈值: {S_BUY})")

    if not is_friday:
        send_pushplus("日间监控报告 - 趋势观察", "<br>".join(logs))
        return

    # 周五正式逻辑
    selected_stocks = []
    if rsrs_signal > S_BUY:
        scores = {}
        for stock in ALPHA_POOL:
            p = get_price_tencent(stock, count=11)
            if len(p) >= 11:
                ret = p['close'].iloc[-1] / p['close'].iloc[0] - 1
                if ret > 0.03: scores[stock] = ret
        if scores:
            best_alpha = sorted(scores, key=scores.get, reverse=True)[0]
            selected_stocks.append(best_alpha)
            logs.append(f"⚔️ 进攻引擎: 选中 {get_stock_display(best_alpha)}")

    beta_scores = {}
    for stock in BETA_POOL:
        p = get_price_tencent(stock, count=21)
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
        p = get_price_tencent(stock, count=21)
        vol = np.std(p['close'].pct_change().dropna())
        vols[stock] = vol if vol > 0 else 0.02
    inv_vol_sum = sum([1.0/v for v in vols.values()])
    for stock in selected_stocks: weights[stock] = (1.0 / vols[stock]) / inv_vol_sum

    # 调仓建议
    logs.append("<h3>🛒 本周正式调仓建议：</h3><ul>")
    for stock, weight in weights.items():
        logs.append(f"<li>标的: <b>{get_stock_display(stock)}</b> | 建议配比: <b>{weight:.2%}</b></li>")
    logs.append("</ul>")

    # 虚拟盘调仓记录
    for stock in list(state['positions'].keys()):
        if stock not in weights:
            state['cash'] += state['positions'][stock] * current_prices[stock]
            del state['positions'][stock]
    for stock, weight in weights.items():
        curr_price = current_prices.get(stock, 0)
        if curr_price > 0:
            shares = int(portfolio_value * weight / curr_price) if stock == CASH_CODE else int(portfolio_value * weight / curr_price / 100) * 100
            old_shares = state['positions'].get(stock, 0)
            state['cash'] += old_shares * curr_price
            state['positions'][stock] = shares
            state['cash'] -= shares * curr_price

    # 调仓后更新持仓展示
    new_portfolio_value = state['cash'] + sum(state['positions'].get(c, 0) * current_prices.get(c, 0) for c in state['positions'])
    logs.append("<h3>📋 调仓后最新持仓：</h3>")
    logs.append(build_position_block(state, current_prices, new_portfolio_value))

    with open(STATE_FILE, 'w') as f: json.dump(state, f)
    send_pushplus(f"🚀 周五正式调仓报告 ({datetime.now().strftime('%Y-%m-%d')})", "<br>".join(logs))

if __name__ == "__main__":
    market_trade()

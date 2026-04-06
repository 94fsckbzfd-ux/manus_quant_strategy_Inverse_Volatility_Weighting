# -*- coding: utf-8 -*-
import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ================= 策略配置 =================
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

# ================= 数据获取引擎 =================
def get_price_tencent(code, count):
    """平替聚宽的 get_price，使用腾讯财经API获取日线复权数据"""
    market = 'sh' if str(code).startswith(('5', '6', '11', '588')) else 'sz'
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,{count},qfq"
    
    try:
        resp = requests.get(url, timeout=10).json()
        data_node = resp.get("data", {}).get(f"{market}{code}", {})
        k_list = data_node.get("qfqday", data_node.get("day", []))
        
        if not k_list:
            return pd.DataFrame()
            
        # 腾讯格式: [date, open, close, high, low, vol]
        df = pd.DataFrame(k_list, columns=['date', 'open', 'close', 'high', 'low', 'vol', 'amt', 'turnover'][:len(k_list[0])])
        for col in ['open', 'close', 'high', 'low']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"获取 {code} 数据失败: {e}")
        return pd.DataFrame()

# ================= 微信推送引擎 =================
def send_pushplus(title, content):
    if not PUSHPLUS_TOKEN:
        print("未配置 PUSHPLUS_TOKEN，跳过推送")
        return
    url = 'http://www.pushplus.plus/send'
    data = {"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"}
    requests.post(url, json=data)

# ================= 核心策略逻辑 =================
def get_rsrs_signal():
    """计算 RSRS 信号"""
    prices = get_price_tencent(MARKET_ANCHOR, count=N_DAYS + M_DAYS)
    if len(prices) < N_DAYS + M_DAYS: 
        return 0
    
    betas = []
    for i in range(len(prices) - N_DAYS + 1):
        df = prices.iloc[i:i+N_DAYS]
        model = LinearRegression().fit(df['low'].values.reshape(-1, 1), df['high'].values)
        betas.append(model.coef_[0])
        
    mean_beta = np.mean(betas)
    std_beta = np.std(betas)
    return (betas[-1] - mean_beta) / std_beta if std_beta != 0 else 0

def market_trade():
    # 1. 加载或初始化本地虚拟账户状态
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        state = {
            "cash": INITIAL_CASH, "positions": {}, 
            "peak_value": INITIAL_CASH, 
            "is_cooling": False, "cool_down_weeks": 0
        }

    # 获取当前各持仓最新价，计算当前净值
    current_prices = {}
    all_codes = set(ALPHA_POOL + BETA_POOL + [CASH_CODE] + list(state['positions'].keys()))
    for code in all_codes:
        df = get_price_tencent(code, 1)
        current_prices[code] = df['close'].iloc[-1] if not df.empty else 0

    portfolio_value = state['cash'] + sum(state['positions'].get(c, 0) * current_prices.get(c, 0) for c in state['positions'])
    
    # 构建推送日志
    logs = [f"📊 当前账户总净值: ¥{portfolio_value:,.2f} (峰值: ¥{state['peak_value']:,.2f})"]

    # --- A. 风控维护 ---
    if portfolio_value > state['peak_value']:
        state['peak_value'] = portfolio_value
    
    current_drawdown = (portfolio_value - state['peak_value']) / state['peak_value'] if state['peak_value'] > 0 else 0
    logs.append(f"📉 当前回撤: {current_drawdown:+.2%}")

    if state['is_cooling']:
        state['cool_down_weeks'] -= 1
        if state['cool_down_weeks'] <= 0:
            state['is_cooling'] = False
            state['peak_value'] = portfolio_value
            logs.append("❄️ 冷静期结束，系统尝试重启！")
        else:
            logs.append(f"❄️ 系统仍在冷静期，剩余 {state['cool_down_weeks']} 周。保持空仓状态。")
            with open(STATE_FILE, 'w') as f: json.dump(state, f)
            send_pushplus(f"量化执行报告 - 剩余冷静期{state['cool_down_weeks']}周", "<br>".join(logs))
            return

    if current_drawdown <= -MAX_DRAWDOWN_LIMIT:
        logs.append(f"🚨 触发硬止损 (回撤≤-15%)，清空所有持仓，进入2周冷静期！")
        state['is_cooling'], state['cool_down_weeks'] = True, 2
        # 清仓处理
        for stock in list(state['positions'].keys()):
            state['cash'] += state['positions'][stock] * current_prices[stock]
        state['positions'] = {}
        
        with open(STATE_FILE, 'w') as f: json.dump(state, f)
        send_pushplus("🚨 触发策略硬止损！", "<br>".join(logs))
        return

    # --- B. 核心逻辑：选股与波动率计算 ---
    rsrs_signal = get_rsrs_signal()
    logs.append(f"📡 沪深300 RSRS信号值: {rsrs_signal:.4f} (阈值: {S_BUY})")
    
    selected_stocks = []
    
    # 1. 进攻引擎 (Alpha)
    if rsrs_signal > S_BUY:
        scores = {}
        for stock in ALPHA_POOL:
            p = get_price_tencent(stock, count=11)
            if len(p) >= 11:
                ret = p['close'].iloc[-1] / p['close'].iloc[0] - 1
                if ret > 0.03: 
                    scores[stock] = ret
        if scores:
            best_alpha = sorted(scores, key=scores.get, reverse=True)[0]
            selected_stocks.append(best_alpha)
            logs.append(f"⚔️ 进攻引擎触发: 选中 {best_alpha} (11日收益 {scores[best_alpha]:.2%})")
            
    # 2. 防御引擎 (Beta)
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
        logs.append(f"🛡️ 防御引擎触发: 选中 {best_beta} (收益波动比 {beta_scores[best_beta]:.4f})")

    if not selected_stocks:
        selected_stocks = [CASH_CODE]
        logs.append(f"💤 无进攻防守信号，防守至现金 {CASH_CODE}")

    # --- C. 核心逻辑：波动率倒数加权 ---
    weights = {}
    vols = {}
    for stock in selected_stocks:
        p = get_price_tencent(stock, count=21)
        vol = np.std(p['close'].pct_change().dropna())
        vols[stock] = vol if vol > 0 else 0.02
        
    inv_vol_sum = sum([1.0/v for v in vols.values()])
    for stock in selected_stocks:
        weights[stock] = (1.0 / vols[stock]) / inv_vol_sum
        
    # --- D. 生成调仓建议并更新虚拟盘 ---
    logs.append("<h3>🛒 本周目标仓位建议：</h3><ul>")
    for stock, weight in weights.items():
        logs.append(f"<li>代码: <b>{stock}</b> | 建议配比: <b>{weight:.2%}</b></li>")
    logs.append("</ul>")

    # 执行虚拟盘调仓记录 (方便下周计算回撤)
    # 1. 卖出不在 target 中的股票
    for stock in list(state['positions'].keys()):
        if stock not in weights:
            state['cash'] += state['positions'][stock] * current_prices[stock]
            del state['positions'][stock]
            
    # 2. 调整目标持仓
    for stock, weight in weights.items():
        curr_price = current_prices.get(stock, 0)
        if curr_price > 0:
            if stock == CASH_CODE:
                shares = int(portfolio_value * weight / curr_price)
            else:
                shares = int(portfolio_value * weight / curr_price / 100) * 100
            
            if shares > 0:
                # 简单处理：先全部折现计算，然后按目标股数全买入（模拟）
                old_shares = state['positions'].get(stock, 0)
                state['cash'] += old_shares * curr_price
                state['positions'][stock] = shares
                state['cash'] -= shares * curr_price

    # 状态持久化写入
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
        
    send_pushplus(f"量化策略自动执行完毕 ({datetime.now().strftime('%Y-%m-%d')})", "<br>".join(logs))
    print("运行成功，状态已保存，推送完毕。")

if __name__ == "__main__":
    market_trade()

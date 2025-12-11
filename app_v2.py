import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px

# --- 1. 設定網頁標題與版面 ---
st.set_page_config(page_title="智能投資組合優化器", layout="wide")
st.title('📈 智能投資組合優化器 (專業銷售版)')
st.markdown("""
此工具提供華爾街等級的投資組合分析：
1. **🛡️ 最小風險 (GMV)**：極致抗跌的保守配置。
2. **🚀 最大夏普 (Max Sharpe)**：追求最高 CP 值的成長配置。
3. **🔥 相關性熱圖**：視覺化展示資產分散效果。
""")

# --- 2. 參數設定 (側邊欄) ---
st.sidebar.header('參數設定')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()
years = st.sidebar.slider('回測年數', 1, 20, 10)
risk_free_rate = 0.02 # 假設無風險利率 2%

# --- 3. 核心邏輯區 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的以進行配置。")
    else:
        with st.spinner('正在連線全球資料庫進行運算...'):
            try:
                # ==========================
                # A. 數據下載與清洗
                # ==========================
                end_date = datetime.today()
                start_date = end_date - timedelta(days=365*years)
                
                # 下載資料
                data = yf.download(user_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                # 處理 yfinance 可能回傳多層索引的問題
                if 'Close' in data.columns:
                    df_close = data['Close']
                else:
                    df_close = data
                
                # 移除空值 (避免停牌或資料不足的影響)
                df_close.dropna(inplace=True)
                
                if df_close.empty:
                    st.error("無法抓取有效數據，請檢查代號或縮短回測年限。")
                    st.stop()

                # 取得最終有效的股票代號列表
                tickers = df_close.columns.tolist()
                
                # 計算基礎統計數據
                returns = df_close.pct_change().dropna()
                cov_matrix = returns.cov() * 252       # 年化共變異數
                mean_returns = returns.mean() * 252    # 年化平均報酬
                corr_matrix = returns.corr()           # 相關係數矩陣
                
                # 優化器的基礎設定
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # 權重總和 = 1
                bounds = tuple((0, 1) for _ in range(num_assets))              # 不做空 (0~1)
                init_guess = [1/num_assets] * num_assets                       # 初始猜測：平均分配
                
                # 準備回測用的「歸一化股價」 (起點都設為 1)
                normalized_prices = df_close / df_close.iloc[0]

                st.success("分析完成！請查看下方詳細報告。")

                # ==========================
                # B. 建立策略分頁
                # ==========================
                tab1, tab2 = st.tabs(["🛡️ 最小風險組合 (保守)", "🚀 最大夏普值組合 (積極)"])

                # --- Tab 1: 最小風險組合 ---
                with tab1:
                    st.subheader("🛡️ 策略目標：極致抗跌，波動最小化")
                    
                    # 1. 執行優化
                    def min_variance(weights, cov_matrix):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    res_min = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                       method='SLSQP', bounds=bounds, constraints=constraints)
                    w_min = res_min.x
                    
                    # 2. 計算預期指標
                    exp_ret_min = np.sum(mean_returns * w_min)
                    exp_vol_min = res_min.fun
                    
                    # 3. 顯示結果 (左欄：配置與指標 / 右欄：回測圖表)
                    col1_1, col1_2 = st.columns([1, 2])
                    
                    with col1_1:
                        st.markdown("### 📊 預期績效")
                        # 使用 columns(2) 分兩欄顯示，避免太擠
                        c1, c2 = st.columns(2)
                        c1.metric("預期年化報酬", f"{exp_ret_min:.2%}")
                        c2.metric("預期年化波動", f"{exp_vol_min:.2%}", delta="極低", delta_color="normal")
                        st.caption("註：基於歷史數據之理論估值")
                        
                        st.divider() # 分隔線
                        
                        # 整理權重表格
                        clean_w = [round(w, 4) if w > 0.0001 else 0.0 for w in w_min]
                        df_min = pd.DataFrame({'標的': tickers, '配置': clean_w})
                        df_min['顯示權重'] = df_min['配置'].apply(lambda x: f"{x:.1%}")

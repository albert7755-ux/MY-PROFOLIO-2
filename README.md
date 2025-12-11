import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px

# --- 1. 設定網頁標題 ---
st.set_page_config(page_title="智能投資組合優化器", layout="wide")
st.title('📈 智能投資組合優化器 (雙策略旗艦版)')
st.markdown("""
此工具提供兩種專業模型分析：
1. **🛡️ 最小風險組合 (GMV)**：追求極致的波動最小化，適合保守型投資人。
2. **🚀 最大夏普值組合 (Max Sharpe)**：追求「性價比 (CP值)」最高，適合追求成長的投資人。
""")

# --- 2. 參數設定 ---
st.sidebar.header('參數設定')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()
years = st.sidebar.slider('回測年數', 1, 20, 10)
risk_free_rate = 0.02 # 假設無風險利率為 2% (用於計算夏普值)

# --- 3. 核心邏輯 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的。")
    else:
        with st.spinner('正在下載全球數據並進行雙重模型運算...'):
            try:
                # ==========================
                # A. 數據準備 (共用區)
                # ==========================
                end_date = datetime.today()
                start_date = end_date - timedelta(days=365*years)
                
                # 下載
                data = yf.download(user_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                if 'Close' in data.columns:
                    df_close = data['Close']
                else:
                    df_close = data
                
                df_close.dropna(inplace=True)
                
                if df_close.empty:
                    st.error("無法抓取數據，請檢查代號。")
                    st.stop()

                tickers = df_close.columns.tolist()
                
                # 計算基礎統計量
                returns = df_close.pct_change().dropna()
                cov_matrix = returns.cov() * 252
                mean_returns = returns.mean() * 252
                
                num_assets = len(tickers)
                
                # 準備共用的限制條件
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets

                # 準備回測用的正規化股價
                normalized_prices = df_close / df_close.iloc[0]

                st.success("數據下載完成！請點選下方分頁切換不同策略。")

                # ==========================
                # B. 建立分頁 (Tabs)
                # ==========================
                tab1, tab2 = st.tabs(["🛡️ 最小風險組合 (保守)", "🚀 最大夏普值組合 (積極)"])

                # ==========================
                # Tab 1: 最小風險 (原本的邏輯)
                # ==========================
                with tab1:
                    st.subheader("🛡️ 策略目標：無論報酬多少，只要波動越小越好")
                    
                    # 1. 運算
                    def min_variance(weights, cov_matrix):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    res_min = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                       method='SLSQP', bounds=bounds, constraints=constraints)
                    
                    # 2. 數據整理
                    w_min = res_min.x
                    exp_ret_min = np.sum(mean_returns * w_min)
                    exp_vol_min = res_min.fun
                    
                    # 3. 顯示結果
                    col1_1, col1_2 = st.columns([1, 2])
                    
                    with col1_1:
                        # 製作表格
                        clean_w = [round(w, 4) if w > 0.0001 else 0.0 for w in w_min]
                        df_min = pd.DataFrame({'標的': tickers, '配置': clean_w})
                        df_min['顯示權重'] = df_min['配置'].apply(lambda x: f"{x:.1%}")
                        df_min = df_min.sort_values('配置', ascending=False)
                        
                        st.info(f"主力配置：**{df_min.iloc[0]['標的']}**")
                        st.table(df_min[['標的', '顯示權重']])
                        
                        st.metric("預期年化報酬", f"{exp_ret_min:.2%}")
                        st.metric("預期年化波動 (風險)", f"{exp_vol_min:.2%}", delta="極低", delta_color="normal")

                        # 圓餅圖
                        fig_pie = px.pie(df_min[df_min['配置']>0], values='配置', names='標的', hole=0.4, title="保守型配置")
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col1_2:
                        # 回測畫圖
                        port_val = (normalized_prices * w_min).sum(axis=1)
                        port_val.name = "🛡️ 最小風險組合"
                        combined = normalized_prices.copy()
                        combined["🛡️ 最小風險組合"] = port_val
                        
                        fig_line = px.line(combined, title=f'資產成長回測 (過去 {years} 年)')
                        fig_line.update_traces(line=dict(width=1), opacity=0.5)
                        fig_line.update_traces(selector=dict(name="🛡️ 最小風險組合"), line=dict(color='green', width=4), opacity=1)
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                        st.metric("期間總報酬率", f"{(port_val.iloc[-1]-1):.2%}")

                # ==========================
                # Tab 2: 最大夏普值 (新功能)
                # ==========================
                with tab2:
                    st.subheader("🚀 策略目標：承擔一分風險，要換回最多報酬 (CP值最高)")
                    st.caption(f"註：假設無風險利率為 {risk_free_rate:.1%}")

                    # 1. 運算 (目標是最小化「負的夏普值」)
                    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
                        p_ret = np.sum(mean_returns * weights)
                        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return - (p_ret - rf) / p_vol
                    
                    args = (mean_returns, cov_matrix, risk_free_rate)
                    res_sharpe = minimize(neg_sharpe_ratio, init_guess, args=args,
                                          method='SLSQP', bounds=bounds, constraints=constraints)
                    
                    # 2. 數據整理
                    w_sharpe = res_sharpe.x
                    exp_ret_sharpe = np.sum(mean_returns * w_sharpe)
                    exp_vol_sharpe = np.sqrt(np.dot(w_sharpe.T, np.dot(cov_matrix, w_sharpe)))
                    sharpe_ratio = (exp_ret_sharpe - risk_free_rate) / exp_vol_sharpe

                    # 3. 顯示結果
                    col2_1, col2_2 = st.columns([1, 2])
                    
                    with col2_1:
                        clean_w_s = [round(w, 4) if w > 0.0001 else 0.0 for w in w_sharpe]
                        df_sharpe = pd.DataFrame({'標的': tickers, '配置': clean_w_s})
                        df_sharpe['顯示權重'] = df_sharpe['配置'].apply(lambda x: f"{x:.1%}")
                        df_sharpe = df_sharpe.sort_values('配置', ascending=False)
                        
                        st.info(f"主力配置：**{df_sharpe.iloc[0]['標的']}**")
                        st.table(df_sharpe[['標的', '顯示權重']])
                        
                        st.metric("預期年化報酬", f"{exp_ret_sharpe:.2%}", delta="較高")
                        st.metric("預期年化波動 (風險)", f"{exp_vol_sharpe:.2%}")
                        st.metric("夏普值 (CP值)", f"{sharpe_ratio:.2f}")

                        # 圓餅圖
                        fig_pie_s = px.pie(df_sharpe[df_sharpe['配置']>0], values='配置', names='標的', hole=0.4, title="積極型配置")
                        st.plotly_chart(fig_pie_s, use_container_width=True)

                    with col2_2:
                        # 回測畫圖
                        port_val_s = (normalized_prices * w_sharpe).sum(axis=1)
                        port_val_s.name = "🚀 最大夏普組合"
                        combined_s = normalized_prices.copy()
                        combined_s["🚀 最大夏普組合"] = port_val_s
                        
                        fig_line_s = px.line(combined_s, title=f'資產成長回測 (過去 {years} 年)')
                        fig_line_s.update_traces(line=dict(width=1), opacity=0.5)
                        fig_line_s.update_traces(selector=dict(name="🚀 最大夏普組合"), line=dict(color='red', width=4), opacity=1)
                        st.plotly_chart(fig_line_s, use_container_width=True)
                        
                        st.metric("期間總報酬率", f"{(port_val_s.iloc[-1]-1):.2%}")

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
else:
    st.info("請輸入代號並開始計算")

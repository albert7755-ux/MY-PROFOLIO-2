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
                    df_min = df_min.sort_values('配置', ascending=False)
                    
                    st.info(f"主力配置：**{df_min.iloc[0]['標的']}**")
                    st.table(df_min[['標的', '顯示權重']])
                    
                    # 圓餅圖
                    fig_pie = px.pie(df_min[df_min['配置']>0], values='配置', names='標的', hole=0.4)
                    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col1_2:
                    # 計算回測淨值曲線 (Buy & Hold)
                    port_val = (normalized_prices * w_min).sum(axis=1)
                    port_val.name = "🛡️ 最小風險組合"
                    combined = normalized_prices.copy()
                    combined["🛡️ 最小風險組合"] = port_val
                    
                    fig_line = px.line(combined, title=f'資產成長回測 (過去 {years} 年)')
                    # 設定線條樣式：組合線加粗，個股線變細且半透明
                    fig_line.update_traces(line=dict(width=1), opacity=0.3)
                    fig_line.update_traces(selector=dict(name="🛡️ 最小風險組合"), line=dict(color='green', width=4), opacity=1)
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    # 計算回測績效 (總報酬 & CAGR)
                    total_ret = port_val.iloc[-1] - 1
                    cagr = (port_val.iloc[-1])**(1/years) - 1
                    
                    st.markdown("### 💰 實際回測結果")
                    cb1, cb2 = st.columns(2)
                    cb1.metric("期間總報酬率", f"{total_ret:.2%}")
                    cb2.metric("回測年化報酬 (CAGR)", f"{cagr:.2%}", help="這段期間平均每年的複利成長率")

            # --- Tab 2: 最大夏普組合 ---
            with tab2:
                st.subheader("🚀 策略目標：承擔一分風險，換取最大報酬 (最高 CP 值)")
                
                # 1. 執行優化 (最小化「負的夏普值」)
                def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
                    p_ret = np.sum(mean_returns * weights)
                    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return - (p_ret - rf) / p_vol
                
                args = (mean_returns, cov_matrix, risk_free_rate)
                res_sharpe = minimize(neg_sharpe_ratio, init_guess, args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                w_sharpe = res_sharpe.x
                
                # 2. 計算預期指標
                exp_ret_sharpe = np.sum(mean_returns * w_sharpe)
                exp_vol_sharpe = np.sqrt(np.dot(w_sharpe.T, np.dot(cov_matrix, w_sharpe)))
                sharpe_ratio = (exp_ret_sharpe - risk_free_rate) / exp_vol_sharpe

                # 3. 顯示結果
                col2_1, col2_2 = st.columns([1, 2])
                
                with col2_1:
                    st.markdown("### 📊 預期績效")
                    
                    # 排版修正：分兩排顯示
                    c_s1, c_s2 = st.columns(2)
                    c_s1.metric("預期年化報酬", f"{exp_ret_sharpe:.2%}")
                    c_s2.metric("預期年化波動", f"{exp_vol_sharpe:.2%}")
                    
                    st.metric("夏普值 (CP值)", f"{sharpe_ratio:.2f}", delta="優異")
                    st.caption("註：基於歷史數據之理論估值")
                    
                    st.divider()

                    clean_w_s = [round(w, 4) if w > 0.0001 else 0.0 for w in w_sharpe]
                    df_sharpe = pd.DataFrame({'標的': tickers, '配置': clean_w_s})
                    df_sharpe['顯示權重'] = df_sharpe['配置'].apply(lambda x: f"{x:.1%}")
                    df_sharpe = df_sharpe.sort_values('配置', ascending=False)
                    
                    st.info(f"主力配置：**{df_sharpe.iloc[0]['標的']}**")
                    st.table(df_sharpe[['標的', '顯示權重']])
                    
                    fig_pie_s = px.pie(df_sharpe[df_sharpe['配置']>0], values='配置', names='標的', hole=0.4)
                    fig_pie_s.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie_s, use_container_width=True)

                with col2_2:
                    port_val_s = (normalized_prices * w_sharpe).sum(axis=1)
                    port_val_s.name = "🚀 最大夏普組合"
                    combined_s = normalized_prices.copy()
                    combined_s["🚀 最大夏普組合"] = port_val_s
                    
                    fig_line_s = px.line(combined_s, title=f'資產成長回測 (過去 {years} 年)')
                    fig_line_s.update_traces(line=dict(width=1), opacity=0.3)
                    fig_line_s.update_traces(selector=dict(name="🚀 最大夏普組合"), line=dict(color='red', width=4), opacity=1)
                    st.plotly_chart(fig_line_s, use_container_width=True)
                    
                    total_ret_s = port_val_s.iloc[-1] - 1
                    cagr_s = (port_val_s.iloc[-1])**(1/years) - 1
                    
                    st.markdown("### 💰 實際回測結果")
                    csb1, csb2 = st.columns(2)
                    csb1.metric("期間總報酬率", f"{total_ret_s:.2%}")
                    csb2.metric("回測年化報酬 (CAGR)", f"{cagr_s:.2%}", help="這段期間平均每年的複利成長率")

            # ==========================
            # C. 進階分析：相關係數矩陣
            # ==========================
            st.markdown("---")
            with st.expander("📊 進階分析：資產相關性熱力圖 (Correlation Heatmap)", expanded=True):
                st.markdown("""
                **如何解讀這張圖？**
                * **紅色 (接近 1)**：兩者走勢高度同步，風險無法分散。
                * **藍色 (接近 0 或負數)**：兩者走勢不相關或相反，**這是資產配置的最佳搭擋！**
                """)
                fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", 
                                     color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("請在左側輸入股票代號並按下「開始計算」")

# --- 4. 免責聲明 (側邊欄底部) ---
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **免責聲明**")
st.sidebar.caption("""
本工具僅供市場分析與模擬參考，不構成任何投資建議或邀約。
歷史績效不代表未來獲利保證。
投資人應審慎評估風險，並自負盈虧。
""")

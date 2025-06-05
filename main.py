import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from model import train_lstm_model, predict_future_prices, evaluate_model
import numpy as np
import ssl
import time
import os

# SSL configuration
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Initialize session state for chart refresh
if 'refresh_charts' not in st.session_state:
    st.session_state.refresh_charts = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Stock Prediction"
if 'selected_lookup_stock' not in st.session_state:
    st.session_state.selected_lookup_stock = "AAPL"

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .nav-button {
        width: 100%;
        margin: 0.25rem 0;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f2f6;
        color: #1f77b4;
        padding: 0.5rem;
        font-weight: bold;
    }
    .nav-button-active {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Custom navigation sidebar
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Prediction", key="nav_prediction", use_container_width=True):
            st.session_state.current_page = "Stock Prediction"
            st.rerun()
    with col2:
        if st.button("🔍 Lookup", key="nav_lookup", use_container_width=True):
            st.session_state.current_page = "Stock Lookup"
            st.rerun()
    
    st.markdown("---")

# Main content based on navigation
if st.session_state.current_page == "Stock Prediction":
     st.sidebar.header("🔧 Configuration")
     st.sidebar.markdown("---")

     st.sidebar.subheader("📊 Stock Selection")
     ticker_input = st.sidebar.text_input(
         "Stock Ticker",
         value="AAPL"
     ).upper().strip()

     st.sidebar.subheader("📅 Date Range")
     col1, col2 = st.sidebar.columns(2)
     with col1:
         start_date = st.date_input(
             "Start Date",
             value=datetime.now() - timedelta(days=730),
             max_value=datetime.now() - timedelta(days=100)
         )
     with col2:
         end_date = st.date_input(
             "End Date",
             value=datetime.now(),
             min_value=start_date + timedelta(days=100),
             max_value=datetime.now()
         )

     st.sidebar.subheader("🔮 Prediction Settings")
     prediction_days = st.sidebar.selectbox(
         "Forecast Horizon",
         options=[7, 14, 30, 60],
         index=2
     )
     time_steps = st.sidebar.selectbox(
         "Lookback Period",
         options=[30, 60, 90],
         index=1
     )

     # Add this option
     use_saved_model = st.sidebar.checkbox("Use saved model (for consistent predictions)", value=False)

     st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
     st.markdown("""
     <div style="text-align: center; margin-bottom: 2rem;">
         <p style="font-size: 1.2rem; color: #666;">
             Powered by LSTM Neural Networks
         </p>
     </div>
     """, unsafe_allow_html=True)

     if not ticker_input:
         st.warning("⚠️ Please enter a stock ticker symbol.")
         st.stop()
     if start_date >= end_date:
         st.error("❌ Start date must be before end date.")
         st.stop()

     @st.cache_data
     def fetch_stock_data(ticker, start, end, max_retries=3):
         for attempt in range(max_retries):
             try:
                 data = yf.Ticker(ticker).history(start=start, end=end, interval='1d', auto_adjust=False)
                 if not data.empty:
                     data.index = pd.to_datetime(data.index)
                     required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                     if not all(col in data.columns for col in required_cols):
                         raise ValueError("Missing required columns in data")
                     for col in ['Open', 'High', 'Low']:
                         if data[col].isnull().all():
                             data[col] = data['Close']
                     return data
                 raise ValueError("Empty data returned")
             except Exception as e:
                 print(f"Attempt {attempt + 1} failed: {str(e)}")
                 time.sleep(1)
         raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts")

     if st.sidebar.button("🚀 Start Analysis", type="primary", use_container_width=True):
         # Reset charts on new analysis
         st.session_state.refresh_charts = True
         progress_container = st.container()
         with progress_container:
             progress_bar = st.progress(0)
             status_text = st.empty()
         try:
             status_text.text("📥 Fetching stock data...")
             progress_bar.progress(20)
             stock_data = fetch_stock_data(ticker_input, start_date, end_date)
             
             if stock_data is None or stock_data.empty:
                 progress_container.empty()
                 st.error(
                     f"❌ No data found for ticker '{ticker_input}' from {start_date} to {end_date}."
                 )
                 st.stop()
             if '20_MA' not in stock_data.columns:
                 stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean().fillna(method='bfill')
             status_text.text("🔍 Validating data...")
             progress_bar.progress(40)
             required_days = time_steps + 50
             if len(stock_data) < required_days:
                 progress_container.empty()
                 st.error(f"❌ Insufficient data. Found {len(stock_data)} days, need at least {required_days} days.")
                 st.stop()
             status_text.text("🧠 Training AI model...")
             progress_bar.progress(60)
             
             # Train new model (removed saved model functionality for now)
             model, scaler, X_test, y_test, df_clean, feature_columns, history = train_lstm_model(
                 stock_data, time_steps=time_steps
             )
             
             status_text.text("🔮 Generating predictions...")
             progress_bar.progress(80)
             future_predictions, future_dates = predict_future_prices(
                 model, scaler, stock_data, feature_columns, days=prediction_days, time_steps=time_steps
             )
             future_predictions = np.nan_to_num(future_predictions, nan=0.0, posinf=0.0, neginf=0.0).astype(float).flatten()
             status_text.text("📊 Evaluating performance...")
             progress_bar.progress(90)
             metrics = evaluate_model(model, scaler, X_test, y_test, feature_columns)
             
             mae = metrics.get("mae", 0.0)
             rmse = metrics.get("rmse", 0.0)
             mape = metrics.get("mape", 0.0)
             r2 = metrics.get("r2", 0.0)
             test_predictions = np.array(metrics.get("test_pred_inverse", []))
             test_actual = np.array(metrics.get("test_actual_inverse", []))

             progress_bar.progress(100)
             time.sleep(0.5)
             progress_container.empty()
             st.success("✅ Analysis completed successfully!")

             # Ensure future_dates are unique and pandas.Timestamp
             future_dates = pd.to_datetime(future_dates)
             if len(future_dates) != len(set(future_dates)) or len(future_dates) != len(future_predictions):
                 future_dates = pd.date_range(start=stock_data.index[-1] + timedelta(days=1), 
                                             periods=len(future_predictions), freq="B")
             
             # Ensure test data alignment
             test_dates = stock_data.index[-len(test_actual):] if test_actual.size > 0 else []

             col1, col2 = st.columns([3, 1])
             with col1:
                 # Main Plotly Chart
                 fig = make_subplots(
                     rows=2, cols=1,
                     shared_xaxes=True,
                     subplot_titles=("Stock Price Prediction", "Trading Volume"),
                     vertical_spacing=0.1,
                     row_heights=[0.7, 0.3]
                 )
                 fig.add_trace(
                     go.Scatter(
                         x=stock_data.index,
                         y=stock_data['Close'],
                         mode='lines',
                         name='Historical Price',
                         line=dict(color='#1f77b4', width=2)
                     ),
                     row=1, col=1
                 )
                 if '20_MA' in stock_data.columns:
                     fig.add_trace(
                         go.Scatter(
                             x=stock_data.index,
                             y=stock_data['20_MA'],
                             mode='lines',
                             name='20-Day MA',
                             line=dict(color='orange', dash='dot'),
                             opacity=0.7
                         ),
                         row=1, col=1
                     )
                 if len(future_dates) == len(future_predictions):
                     fig.add_trace(
                         go.Scatter(
                             x=future_dates,
                             y=future_predictions,
                             mode='lines+markers',
                             name='Predicted Price',
                             line=dict(color='red', dash='dash', width=3),
                             marker=dict(size=6)
                         ),
                         row=1, col=1
                     )
                     # Add prediction confidence interval
                     std_dev = np.std(test_actual - test_predictions) if test_actual.size > 0 else np.std(future_predictions)
                     upper_bound = np.array(future_predictions) + 2 * std_dev
                     lower_bound = np.array(future_predictions) - 2 * std_dev
                     fig.add_trace(
                         go.Scatter(
                             x=list(future_dates) + list(future_dates)[::-1],
                             y=np.concatenate([upper_bound, lower_bound[::-1]]),
                             fill='toself',
                             fillcolor='rgba(255,0,0,0.1)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name='Confidence Interval',
                             showlegend=True
                         ),
                         row=1, col=1
                     )
                 fig.add_trace(
                     go.Bar(
                         x=stock_data.index,
                         y=stock_data['Volume'],
                         name='Volume',
                         marker_color='lightblue',
                         opacity=0.6
                     ),
                     row=2, col=1
                 )
                 fig.update_layout(
                     height=700,
                     title=f"{ticker_input} Stock Analysis & Prediction",
                     xaxis_title="Date",
                     yaxis_title="Price (USD)",
                     yaxis2_title="Volume",
                     hovermode='x unified',
                     legend=dict(x=0, y=1),
                     template="plotly_white"
                 )
                 fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
                 st.plotly_chart(fig, use_container_width=True)
             
             with col2:
                 st.subheader("📈 Model Performance")
                 st.table({
                     "MAE ($)": [f"{float(mae):.2f}"],
                     "RMSE ($)": [f"{float(rmse):.2f}"],
                     "MAPE (%)": [f"{float(mape):.2f}"],
                     "R²": [f"{float(r2):.3f}"]
                 })
                 st.subheader("🔮 Prediction Summary")
                 current_price = float(stock_data['Close'].iloc[-1])
                 predicted_price = float(future_predictions[-1]) if len(future_predictions) > 0 else float('nan')
                 price_change = predicted_price - current_price
                 price_change_pct = (price_change / current_price) * 100 if current_price != 0 else 0
                 st.metric(
                     label="Current Price",
                     value=f"${current_price:.2f}"
                 )
                 st.metric(
                     label=f"Predicted Price ({prediction_days}d)",
                     value=f"${predicted_price:.2f}",
                     delta=f"{price_change_pct:+.1f}%"
                 )
                 st.subheader("💡 AI Recommendation")
                 if price_change_pct > 5:
                     st.success("🟢 **STRONG BUY**\nModel predicts significant upward movement")
                 elif price_change_pct > 2:
                     st.info("🔵 **BUY**\nModel predicts moderate upward movement")
                 elif price_change_pct > -2:
                     st.warning("🟡 **HOLD**\nModel predicts sideways movement")
                 elif price_change_pct > -5:
                     st.warning("🟠 **SELL**\nModel predicts moderate downward movement")
                 else:
                     st.error("🔴 **STRONG SELL**\nModel predicts significant downward movement")
             
             with st.expander("📋 Data Summary"):
                 col1, col2, col3 = st.columns(3)
                 with col1:
                     st.write("**Dataset Info**")
                     st.write(f"• Total records: {len(stock_data)}")
                     st.write(f"• Training period: {(end_date - start_date).days} days")
                     st.write(f"• Features used: {len(feature_columns)}")
                 with col2:
                     st.write("**Price Statistics**")
                     st.write(f"• Highest: ${stock_data['High'].max():.2f}")
                     st.write(f"• Lowest: ${stock_data['Low'].min():.2f}")
                     st.write(f"• Average: ${stock_data['Close'].mean():.2f}")
                 with col3:
                     st.write("**Model Info**")
                     st.write(f"• Lookback period: {time_steps} days")
                     st.write(f"• Prediction horizon: {prediction_days} days")
                     st.write(f"• Training epochs: {len(history.history['loss'])}")
             
             with st.expander("📊 Recent Data"):
                 recent_data = stock_data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
                 st.dataframe(recent_data.style.format({
                     'Open': '${:.2f}',
                     'High': '${:.2f}',
                     'Low': '${:.2f}',
                     'Close': '${:.2f}',
                     'Volume': '{:,.0f}'
                 }))
             
             pred_df = pd.DataFrame({
                 "Date": future_dates,
                 "Predicted_Close": future_predictions
             })
             st.download_button(
                 label="Download Predictions as CSV",
                 data=pred_df.to_csv(index=False),
                 file_name=f"{ticker_input}_predictions.csv",
                 mime="text/csv"
             )
             
             with st.expander("📉 Model Training Loss Curve"):
                 if 'loss' in history.history:
                     st.line_chart(pd.Series(history.history['loss'], name="Training Loss"), 
                                  use_container_width=True)
                 else:
                     st.info("No training loss data available.")
             
             with st.expander("📈 Actual vs Predicted (Test Set)"):
                 if test_actual.size > 0 and test_predictions.size > 0 and len(test_dates) == len(test_actual):
                     st.line_chart({
                         "Actual": pd.Series(test_actual, index=test_dates, name="Actual"),
                         "Predicted": pd.Series(test_predictions, index=test_dates, name="Predicted")
                     }, use_container_width=True)
                 else:
                     st.info("Not enough test data for actual vs predicted plot.")
         
         except Exception as e:
             progress_container.empty()
             st.error(f"❌ An error occurred: {str(e)}")
             st.info("💡 **Troubleshooting tips:**\n"
                     "• Check if the ticker symbol is valid\n"
                     "• Ensure sufficient historical data is available\n"
                     "• Try a different date range\n"
                     "• Check your internet connection")
             if st.button("🔄 Retry Analysis", type="secondary", key="retry"):
                 st.session_state.refresh_charts = True
                 st.rerun()
     else:
         st.info("👆 Configure your settings in the sidebar and click 'Start Analysis' to begin!")
         col1, col2, col3 = st.columns(3)
         with col1:
             st.markdown("""
             ### 🧠 Advanced AI
             - LSTM neural networks
             - Technical indicators
             - Pattern recognition
             """)
         with col2:
             st.markdown("""
             ### 📊 Comprehensive Analysis
             - Historical trends
             - Volume analysis
             - Performance metrics
             """)
         with col3:
             st.markdown("""
             ### 🔮 Future Predictions
             - Multi-day forecasts
             - Confidence intervals
             - Trading recommendations
             """)

elif st.session_state.current_page == "Stock Lookup":
    # Enhanced Stock Lookup Navigation
    with st.sidebar:
        st.header("🔍 Lookup Controls")
        st.markdown("---")
        
        # Quick stock selector
        st.subheader("🚀 Quick Select")
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "UBER", "SPOT"]
        
        selected_stock = st.selectbox(
            "Choose a stock:",
            popular_stocks,
            index=popular_stocks.index(st.session_state.selected_lookup_stock) 
            if st.session_state.selected_lookup_stock in popular_stocks else 0
        )
        
        if selected_stock != st.session_state.selected_lookup_stock:
            st.session_state.selected_lookup_stock = selected_stock
            st.rerun()
        
        # Custom ticker input
        st.subheader("📝 Custom Ticker")
        custom_ticker = st.text_input(
            "Enter ticker:",
            value="",
            placeholder="e.g., TSLA, AMD, CRM"
        ).upper().strip()
        
        if st.button("🔍 Lookup Custom", use_container_width=True) and custom_ticker:
            st.session_state.selected_lookup_stock = custom_ticker
            st.rerun()
        
        # Market sectors
        st.subheader("🏢 By Sector")
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
            "Finance": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
            "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE"]
        }
        
        selected_sector = st.selectbox("Select sector:", list(sectors.keys()))
        sector_stock = st.selectbox("Select stock:", sectors[selected_sector])
        
        if st.button("📊 Analyze Sector Stock", use_container_width=True):
            st.session_state.selected_lookup_stock = sector_stock
            st.rerun()
    
    # Main lookup content
    st.header("🔍 Stock Information Lookup")
    st.subheader(f"Analyzing: {st.session_state.selected_lookup_stock}")
    
    # Auto-fetch data for selected stock
    try:
        with st.spinner(f"Fetching data for {st.session_state.selected_lookup_stock}..."):
            stock = yf.Ticker(st.session_state.selected_lookup_stock)
            info = stock.info
            
            # Enhanced display with tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Charts", "📰 News", "📋 Details"])
            
            with tab1:
                # Key metrics in a more visual layout
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = info.get('currentPrice', 'N/A')
                    prev_close = info.get('previousClose', 'N/A')
                    if current_price != 'N/A' and prev_close != 'N/A':
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100
                        st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
                    else:
                        st.metric("Current Price", f"${current_price}")
                
                with col2:
                    market_cap = info.get('marketCap')
                    if market_cap:
                        if market_cap >= 1e12:
                            display_cap = f"${market_cap/1e12:.1f}T"
                        elif market_cap >= 1e9:
                            display_cap = f"${market_cap/1e9:.1f}B"
                        else:
                            display_cap = f"${market_cap/1e6:.1f}M"
                        st.metric("Market Cap", display_cap)
                    else:
                        st.metric("Market Cap", "N/A")
                
                with col3:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                
                with col4:
                    st.metric("52W High/Low", f"${info.get('fiftyTwoWeekHigh', 'N/A')} / ${info.get('fiftyTwoWeekLow', 'N/A')}")
                
                # Company info
                st.subheader(f"📋 {info.get('longName', st.session_state.selected_lookup_stock)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                
                with col2:
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Employees:** N/A")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
                
                if info.get('longBusinessSummary'):
                    with st.expander("📝 Business Description"):
                        st.write(info.get('longBusinessSummary'))
            
            with tab2:
                # Multiple chart options
                chart_period = st.selectbox("Chart Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
                hist_data = stock.history(period=chart_period)
                
                if not hist_data.empty:
                    # Candlestick chart
                    fig = go.Figure(data=go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close']
                    ))
                    fig.update_layout(
                        title=f"{st.session_state.selected_lookup_stock} - {chart_period.upper()} Candlestick Chart",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    vol_fig = go.Figure()
                    vol_fig.add_trace(go.Bar(
                        x=hist_data.index,
                        y=hist_data['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    vol_fig.update_layout(
                        title="Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        template="plotly_white",
                        height=300
                    )
                    st.plotly_chart(vol_fig, use_container_width=True)
            
            with tab3:
                # Enhanced news section
                try:
                    news = stock.news
                    if news:
                        st.subheader(f"📰 Latest News for {st.session_state.selected_lookup_stock}")
                        for i, article in enumerate(news[:8]):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**{article.get('title', 'N/A')}**")
                                    st.write(f"*{article.get('publisher', 'N/A')}*")
                                with col2:
                                    if article.get('link'):
                                        st.link_button("Read More", article['link'])
                                if i < len(news[:8]) - 1:
                                    st.divider()
                    else:
                        st.info("No recent news available")
                except:
                    st.info("News data unavailable")
            
            with tab4:
                # Detailed financial information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💰 Valuation Metrics")
                    metrics_data = {
                        "Market Cap": f"${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A",
                        "Enterprise Value": f"${info.get('enterpriseValue', 'N/A'):,}" if info.get('enterpriseValue') else "N/A",
                        "Price to Book": info.get('priceToBook', 'N/A'),
                        "Price to Sales": info.get('priceToSalesTrailing12Months', 'N/A'),
                        "PEG Ratio": info.get('pegRatio', 'N/A'),
                        "Beta": info.get('beta', 'N/A')
                    }
                    for metric, value in metrics_data.items():
                        st.write(f"• **{metric}:** {value}")
                
                with col2:
                    st.subheader("📊 Profitability")
                    profit_data = {
                        "Profit Margin": f"{info.get('profitMargins', 'N/A'):.2%}" if info.get('profitMargins') else "N/A",
                        "Operating Margin": f"{info.get('operatingMargins', 'N/A'):.2%}" if info.get('operatingMargins') else "N/A",
                        "Return on Assets": f"{info.get('returnOnAssets', 'N/A'):.2%}" if info.get('returnOnAssets') else "N/A",
                        "Return on Equity": f"{info.get('returnOnEquity', 'N/A'):.2%}" if info.get('returnOnEquity') else "N/A",
                        "Revenue Growth": f"{info.get('revenueGrowth', 'N/A'):.2%}" if info.get('revenueGrowth') else "N/A",
                        "Earnings Growth": f"{info.get('earningsGrowth', 'N/A'):.2%}" if info.get('earningsGrowth') else "N/A"
                    }
                    for metric, value in profit_data.items():
                        st.write(f"• **{metric}:** {value}")
                
                # Additional data tables
                try:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏛️ Top Institutional Holdings")
                        institutional_holders = stock.institutional_holders
                        if institutional_holders is not None and not institutional_holders.empty:
                            st.dataframe(institutional_holders.head(5), use_container_width=True)
                    
                    with col2:
                        st.subheader("🎯 Analyst Recommendations")
                        recommendations = stock.recommendations
                        if recommendations is not None and not recommendations.empty:
                            st.dataframe(recommendations.tail(5), use_container_width=True)
                except:
                    st.info("Additional data unavailable")
            
            # Quick action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔮 Get AI Prediction", key="predict_lookup", use_container_width=True):
                    st.session_state.current_page = "Stock Prediction"
                    # Update the prediction tab's ticker
                    st.session_state.prediction_ticker = st.session_state.selected_lookup_stock
                    st.rerun()
            with col2:
                if st.button("📊 Download Data", key="download_lookup", use_container_width=True):
                    # Download historical data
                    hist_data = stock.history(period="1y")
                    csv = hist_data.to_csv()
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{st.session_state.selected_lookup_stock}_data.csv",
                        mime="text/csv"
                    )
            with col3:
                if st.button("🔄 Refresh Data", key="refresh_lookup", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
                    
    except Exception as e:
        st.error(f"❌ Error fetching data for {st.session_state.selected_lookup_stock}: {str(e)}")
        st.info("Please check if the ticker symbol is valid or try again later.")
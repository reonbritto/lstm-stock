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
import requests
import feedparser

# SSL configuration
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Initialize session state for chart refresh
if 'refresh_charts' not in st.session_state:
    st.session_state.refresh_charts = False

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
</style>
""", unsafe_allow_html=True)

# Navigation
nav = st.sidebar.radio("🔀 Navigation", ["Stock Prediction", "Stock Lookup"])

if nav == "Stock Prediction":
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

elif nav == "Stock Lookup":
    st.markdown('<h1 class="main-header">🔍 Stock Information Center</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Get comprehensive information about any stock
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced lookup form with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lookup_ticker = st.text_input(
            "🔍 Enter Stock Ticker Symbol",
            value="AAPL",
            key="lookup_ticker",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            help="Enter a valid stock ticker symbol"
        ).upper().strip()
        
        lookup_button = st.button("📊 Get Stock Information", type="primary", use_container_width=True)
    if lookup_button:
        if lookup_ticker:
            with st.spinner(f"Fetching information for {lookup_ticker}..."):
                try:
                    # Fetch stock info
                    stock = yf.Ticker(lookup_ticker)
                    info = stock.info
                    
                    # Validate if we got valid data
                    if not info or info.get('symbol') != lookup_ticker:
                        st.error(f"❌ Could not find valid data for '{lookup_ticker}'. Please check the ticker symbol.")
                        st.stop()
                    
                    st.success(f"✅ Successfully loaded data for {info.get('longName', lookup_ticker)}")
                    
                    # Display basic information
                    st.subheader("📊 Key Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                        previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
                        market_cap = info.get('marketCap')
                        
                        if current_price:
                            price_change = current_price - previous_close if previous_close else 0
                            price_change_pct = (price_change / previous_close * 100) if previous_close else 0
                            st.metric("Current Price", f"${current_price:.2f}", 
                                     delta=f"{price_change_pct:+.2f}%" if price_change_pct else None)
                        else:
                            st.metric("Current Price", "N/A")
                            
                        st.metric("Previous Close", f"${previous_close:.2f}" if previous_close else "N/A")
                        st.metric("Market Cap", f"${market_cap:,.0f}" if market_cap else "N/A")
                    
                    with col2:
                        week_high = info.get('fiftyTwoWeekHigh')
                        week_low = info.get('fiftyTwoWeekLow')
                        volume = info.get('volume') or info.get('regularMarketVolume')
                        
                        st.metric("52 Week High", f"${week_high:.2f}" if week_high else "N/A")
                        st.metric("52 Week Low", f"${week_low:.2f}" if week_low else "N/A")
                        st.metric("Volume", f"{volume:,}" if volume else "N/A")
                    
                    with col3:
                        pe_ratio = info.get('trailingPE')
                        dividend_yield = info.get('dividendYield')
                        beta = info.get('beta')
                        
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                        st.metric("Dividend Yield", f"{dividend_yield}%" if dividend_yield else "N/A")   
                        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
                
                    # Company Information
                    with st.expander("📋 Company Information", expanded=True):
                        st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.write(f"**Country:** {info.get('country', 'N/A')}")
                        website = info.get('website')
                        if website:
                            st.write(f"**Website:** [{website}]({website})")
                        else:
                            st.write("**Website:** N/A")
                        
                        summary = info.get('longBusinessSummary')
                        if summary:
                            st.write(f"**Description:** {summary[:500]}..." if len(summary) > 500 else f"**Description:** {summary}")
                        else:
                            st.write("**Description:** N/A")
                
                    # Financial Metrics
                    with st.expander("💰 Financial Metrics"):
                        fin_col1, fin_col2 = st.columns(2)
                        
                        with fin_col1:
                            st.write("**Valuation Metrics:**")
                            market_cap = info.get('marketCap')
                            enterprise_value = info.get('enterpriseValue')
                            price_to_book = info.get('priceToBook')
                            price_to_sales = info.get('priceToSalesTrailing12Months')
                            
                            st.write(f"• Market Cap: ${market_cap:,.0f}" if market_cap else "• Market Cap: N/A")
                            st.write(f"• Enterprise Value: ${enterprise_value:,.0f}" if enterprise_value else "• Enterprise Value: N/A")
                            st.write(f"• Price to Book: {price_to_book:.2f}" if price_to_book else "• Price to Book: N/A")
                            st.write(f"• Price to Sales: {price_to_sales:.2f}" if price_to_sales else "• Price to Sales: N/A")
                        
                        with fin_col2:
                            st.write("**Profitability:**")
                            profit_margin = info.get('profitMargins')
                            operating_margin = info.get('operatingMargins')
                            roa = info.get('returnOnAssets')
                            roe = info.get('returnOnEquity')
                            
                            st.write(f"• Profit Margin: {profit_margin*100:.2f}%" if profit_margin else "• Profit Margin: N/A")
                            st.write(f"• Operating Margin: {operating_margin*100:.2f}%" if operating_margin else "• Operating Margin: N/A")
                            st.write(f"• Return on Assets: {roa*100:.2f}%" if roa else "• Return on Assets: N/A")
                            st.write(f"• Return on Equity: {roe*100:.2f}%" if roe else "• Return on Equity: N/A")
                
                    # Recent Performance Chart
                    with st.expander("📈 Recent Price Chart", expanded=True):
                        # Get 1 year of data for chart
                        hist_data = stock.history(period="1y")
                        if not hist_data.empty:
                            # Create a more detailed chart with candlesticks
                            from plotly.subplots import make_subplots
                            chart_fig = go.Figure()
                            
                            # Add candlestick chart
                            chart_fig.add_trace(go.Candlestick(
                                x=hist_data.index,
                                open=hist_data['Open'],
                                high=hist_data['High'],
                                low=hist_data['Low'],
                                close=hist_data['Close'],
                                name="Price"
                            ))
                            
                            # Add moving averages
                            ma_20 = hist_data['Close'].rolling(window=20).mean()
                            ma_50 = hist_data['Close'].rolling(window=50).mean()
                            
                            chart_fig.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=ma_20,
                                mode='lines',
                                name='20-day MA',
                                line=dict(color='orange', width=1)
                            ))
                            
                            chart_fig.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=ma_50,
                                mode='lines',
                                name='50-day MA',
                                line=dict(color='red', width=1)
                            ))
                            
                            chart_fig.update_layout(
                                title=f"{lookup_ticker} - 1 Year Price Chart",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                template="plotly_white",
                                height=500,
                                xaxis_rangeslider_visible=False
                            )
                            st.plotly_chart(chart_fig, use_container_width=True)
                        else:
                            st.warning("No historical data available for chart.")
                
                    # News (via Yahoo Finance RSS)
                    with st.expander("📰 Recent News"):
                        feed_url = (
                            f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                            f"?s={lookup_ticker}&region=US&lang=en-US"
                        )
                        try:
                            with st.spinner(f"Fetching news for {lookup_ticker}..."):
                                feed = feedparser.parse(feed_url)
                                entries = feed.entries or []
                            
                            if entries:
                                st.subheader(f"Latest {len(entries)} Articles for {info.get('longName', lookup_ticker)}")
                                for i, entry in enumerate(entries, 1):
                                    with st.container():
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            title = entry.get("title", "N/A")
                                            st.markdown(f"**{i}. {title}**")
                                            
                                            published_date = entry.get("published", "No date available")
                                            # Attempt to parse and reformat date for consistency
                                            try:
                                                dt_obj = datetime.strptime(published_date, "%a, %d %b %Y %H:%M:%S %Z")
                                                formatted_date = dt_obj.strftime("%B %d, %Y %H:%M %Z")
                                            except ValueError:
                                                formatted_date = published_date # Keep original if parsing fails
                                            
                                            st.caption(f"Published: {formatted_date}")

                                        link = entry.get("link", "")
                                        if link:
                                            with col2:
                                                st.link_button("Read Article ↗️", link, use_container_width=True)
                                        
                                        summary = entry.get("summary", "")
                                        if summary:
                                            # Basic HTML tag removal for cleaner summary
                                            import re
                                            clean_summary = re.sub('<[^<]+?>', '', summary)
                                            st.markdown(f"<small>{clean_summary[:300]}{'...' if len(clean_summary) > 300 else ''}</small>", unsafe_allow_html=True)
                                        
                                        # Check for media content (thumbnails)
                                        if 'media_content' in entry and entry.media_content:
                                            for media in entry.media_content:
                                                if 'url' in media:
                                                    st.image(media['url'], width=150)
                                                    break # Show first image
                                        elif 'media_thumbnail' in entry and entry.media_thumbnail:
                                             for thumbnail in entry.media_thumbnail:
                                                if 'url' in thumbnail:
                                                    st.image(thumbnail['url'], width=150)
                                                    break # Show first image
                                        
                                        st.divider()
                            else:
                                st.info(f"No recent RSS news found for {lookup_ticker}.")
                        except Exception as e:
                            st.error(f"Could not fetch or parse news feed: {e}")
                
                    # Institutional Holdings
                    with st.expander("🏛️ Institutional Holdings"):
                        try:
                            institutional_holders = stock.institutional_holders
                            if institutional_holders is not None and not institutional_holders.empty:
                                st.dataframe(institutional_holders.head(10), use_container_width=True)
                            else:
                                st.info("Institutional holdings data unavailable")
                        except Exception:
                            st.info("Institutional holdings data unavailable")
                
                    # Recommendations
                    with st.expander("🎯 Analyst Recommendations"):
                        try:
                            recommendations = stock.recommendations
                            if recommendations is not None and not recommendations.empty:
                                st.subheader("Latest Analyst Ratings")
                                
                                # Convert index to DatetimeIndex if it's not already, for proper sorting and formatting
                                if not isinstance(recommendations.index, pd.DatetimeIndex):
                                    recommendations.index = pd.to_datetime(recommendations.index)
                                
                                # Sort by date descending to show latest first
                                recommendations = recommendations.sort_index(ascending=False)
                                
                                # Display summary of recommendations
                                if 'To Grade' in recommendations.columns:
                                    st.markdown("##### Recommendation Summary (Last 12 Months)")
                                    recent_recommendations = recommendations[recommendations.index > (datetime.now() - timedelta(days=365))]
                                    if not recent_recommendations.empty:
                                        grade_counts = recent_recommendations['To Grade'].value_counts()
                                        
                                        # Custom order for grades if needed, otherwise sort by count
                                        grade_order = ['Strong Buy', 'Buy', 'Hold', 'Underperform', 'Sell', 'Strong Sell']
                                        # Filter and reorder grade_counts based on grade_order
                                        filtered_grade_counts = pd.Series(dtype='int')
                                        for grade in grade_order:
                                            if grade in grade_counts.index:
                                                filtered_grade_counts[grade] = grade_counts[grade]
                                        
                                        # Add any grades not in grade_order but present in grade_counts
                                        for grade in grade_counts.index:
                                            if grade not in filtered_grade_counts.index:
                                                filtered_grade_counts[grade] = grade_counts[grade]

                                        if not filtered_grade_counts.empty:
                                            # Display as a table
                                            summary_df = pd.DataFrame(filtered_grade_counts).reset_index()
                                            summary_df.columns = ['Recommendation', 'Count']
                                            st.table(summary_df)

                                            # Optional: Bar chart for visual summary
                                            try:
                                                import plotly.express as px
                                                fig_rec_summary = px.bar(summary_df, x='Recommendation', y='Count', 
                                                                         title="Recommendation Distribution (Last 12 Months)",
                                                                         color='Recommendation',
                                                                         labels={'Count': 'Number of Ratings'})
                                                fig_rec_summary.update_layout(xaxis_title="Rating", yaxis_title="Count")
                                                st.plotly_chart(fig_rec_summary, use_container_width=True)
                                            except ImportError:
                                                st.caption("Install plotly for a visual summary chart.")
                                            except Exception as e:
                                                st.caption(f"Could not generate recommendation chart: {e}")

                                        else:
                                            st.info("No 'To Grade' data found in recent recommendations to summarize.")
                                    else:
                                        st.info("No recommendations in the last 12 months to summarize.")
                                else:
                                    st.info("Recommendation data available, but 'To Grade' column is missing for summary.")

                                st.markdown("##### Detailed Recommendations")
                                # Select and rename columns for better readability
                                display_cols = {
                                    'Firm': 'Analyst Firm',
                                    'To Grade': 'Rating',
                                    'From Grade': 'Previous Rating',
                                    'Action': 'Action'
                                }
                                # Filter columns that exist in the dataframe
                                existing_display_cols = {k: v for k, v in display_cols.items() if k in recommendations.columns}
                                
                                if existing_display_cols:
                                    recommendations_display = recommendations[list(existing_display_cols.keys())].copy()
                                    recommendations_display.rename(columns=existing_display_cols, inplace=True)
                                    # Format the date index
                                    recommendations_display.index = recommendations_display.index.strftime('%Y-%m-%d')
                                    st.dataframe(recommendations_display.head(20), use_container_width=True) # Show top 20
                                else:
                                    st.dataframe(recommendations.head(20), use_container_width=True)


                            else:
                                st.info("Analyst recommendations data unavailable for this stock.")
                        except Exception as e:
                            st.error(f"Could not fetch or display analyst recommendations: {e}")
                
                except Exception as e:
                    st.error(f"❌ Error fetching data for {lookup_ticker}: {str(e)}")
                    st.info("💡 **Tips:**\n"
                           "• Make sure the ticker symbol is correct\n"
                           "• Try a different symbol\n"
                           "• Check your internet connection")
        else:
            st.warning("⚠️ Please enter a stock ticker symbol to continue.")
    
    # Quick lookup buttons for popular stocks
    st.divider()
    st.subheader("🔥 Popular Stocks")
    st.caption("Click on any stock below to quickly look it up")
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    
    cols = st.columns(4)
    for i, stock in enumerate(popular_stocks):
        with cols[i % 4]:
            if st.button(stock, key=f"popular_{stock}"):
                st.session_state.lookup_ticker = stock
                st.rerun()
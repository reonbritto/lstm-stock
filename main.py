import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from model import train_lstm_model, predict_future_prices, evaluate_model
import numpy as np
import ssl
import os
import time
import json
from requests.exceptions import RequestException
import requests

# SSL configuration
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🔧 Configuration")
st.sidebar.markdown("---")

# Stock selection
st.sidebar.subheader("📊 Stock Selection")
ticker_input = st.sidebar.text_input(
    "Stock Ticker", 
    value="AAPL", 
    help="Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)"
).upper().strip()

# Date range selection
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date", 
        value=datetime.now() - timedelta(days=730),  # 2 years of data
        max_value=datetime.now() - timedelta(days=100)
    )
with col2:
    end_date = st.date_input(
        "End Date", 
        value=datetime.now(),
        min_value=start_date + timedelta(days=100),
        max_value=datetime.now()
    )

# Prediction settings
st.sidebar.subheader("🔮 Prediction Settings")
prediction_days = st.sidebar.selectbox(
    "Forecast Horizon", 
    options=[7, 14, 30, 60], 
    index=2,
    help="Number of future days to predict"
)

time_steps = st.sidebar.selectbox(
    "Lookback Period", 
    options=[30, 60, 90], 
    index=1,
    help="Number of historical days to use for prediction"
)

# Advanced settings
st.sidebar.subheader("⚙️ Advanced Settings")
n_trials = st.sidebar.slider(
    "Optuna Trials", 
    min_value=5, 
    max_value=50, 
    value=5,
    help="Number of hyperparameter optimization trials (more = better but slower)"
)

# Main interface
st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Advanced LSTM neural network for stock price forecasting
    </p>
</div>
""", unsafe_allow_html=True)

# Validation
if not ticker_input:
    st.warning("⚠️ Please enter a stock ticker symbol.")
    st.stop()

if start_date >= end_date:
    st.error("❌ Start date must be before end date.")
    st.stop()

# Fetch stock data function
def fetch_stock_data(ticker, start, end, max_retries=3):
    """
    Fetch stock data using yf.Ticker(ticker).history() with retry logic.
    If data is empty, it's likely the ticker is incorrect, delisted, or lacks data.
    """
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, interval='1d', auto_adjust=False)
            if not data.empty:
                data.index = pd.to_datetime(data.index)
                for col in ['Open', 'High', 'Low']:
                    if col not in data.columns or data[col].isnull().all():
                        data[col] = data['Close']
                return data
        except Exception as e:
            print(f"Attempt {attempt+1} to fetch ticker '{ticker}' failed: {e}")
        time.sleep(1)
    return None

# Main prediction button
if st.sidebar.button("🚀 Start Analysis", type="primary", use_container_width=True):
    
    # Create progress container
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Step 1: Fetch data
        status_text.text("📥 Fetching stock data...")
        progress_bar.progress(20)
        
        stock_data = fetch_stock_data(ticker_input, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            progress_container.empty()
            st.error(
                f"❌ No data found for ticker '{ticker_input}' from {start_date} to {end_date}. "
                "Check the symbol, your internet connection, or try a different date range. "
                "The ticker might be delisted or have no available data."
            )
            st.stop()
        
        # Calculate 20_MA if not present
        if '20_MA' not in stock_data.columns:
            stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean().fillna(method='bfill')
        
        # Step 2: Data validation
        status_text.text("🔍 Validating data...")
        progress_bar.progress(40)
        
        required_days = time_steps + 50  # Minimum required data points
        if len(stock_data) < required_days:
            progress_container.empty()
            st.error(f"❌ Insufficient data. Found {len(stock_data)} days, need at least {required_days} days.")
            st.stop()
        
        # Step 3: Train model
        status_text.text("🧠 Training AI model...")
        progress_bar.progress(60)
        
        model, scaler, X_test, y_test, df_clean, feature_columns, history, best_params = train_lstm_model(
            stock_data, time_steps=time_steps, n_trials=n_trials
        )
        
        # Step 4: Make predictions
        status_text.text("🔮 Generating predictions...")
        progress_bar.progress(80)
        
        future_predictions, future_dates = predict_future_prices(
            model, scaler, stock_data, feature_columns, days=prediction_days, time_steps=time_steps
        )
        # Ensure future_predictions is a 1D float array for formatting
        if isinstance(future_predictions, np.ndarray):
            future_predictions = future_predictions.astype(float).flatten()
        else:
            future_predictions = np.array(future_predictions, dtype=float).flatten()
        # Remove nan/inf from predictions for display
        future_predictions = np.nan_to_num(future_predictions, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 5: Evaluate model
        status_text.text("📊 Evaluating performance...")
        progress_bar.progress(90)
        
        metrics = evaluate_model(model, scaler, X_test, y_test, feature_columns)
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        mape = metrics["mape"]
        r2 = metrics["r2"]
        test_predictions = metrics["test_pred_inverse"]
        test_actual = metrics["test_actual_inverse"]
        
        # Clear progress indicators
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_container.empty()
        
        # Display results
        st.success("✅ Analysis completed successfully!")
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Main chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("Stock Price Prediction", "Trading Volume"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Historical prices
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
            
            # Moving average
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
            
            # Future predictions
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
            last_price = stock_data['Close'].iloc[-1]
            std_dev = np.std(test_actual - test_predictions)
            upper_bound = future_predictions + 2 * std_dev
            lower_bound = future_predictions - 2 * std_dev
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Volume
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
            
            # Update layout
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
            # Performance metrics
            st.subheader("📈 Model Performance")
            st.table({
                "MAE ($)": [f"{float(mae):.2f}"],
                "RMSE ($)": [f"{float(rmse):.2f}"],
                "MAPE (%)": [f"{float(mape):.2f}"],
                "R²": [f"{float(r2):.3f}"]
            })
            
            # Prediction summary
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
            
            # Trading recommendation
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
        
        # Additional information
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
                
        # Hyperparameter optimization results
        with st.expander("🔧 Hyperparameter Optimization"):
            st.write("**Best Parameters Found:**")
            for param, value in best_params.items():
                st.write(f"• {param}: {value}")
        
        # Recent data table
        with st.expander("📊 Recent Data"):
            recent_data = stock_data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
            st.dataframe(recent_data.style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }))
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 **Troubleshooting tips:**\n"
                "• Check if the ticker symbol is valid\n"
                "• Ensure sufficient historical data is available\n"
                "• Try a different date range\n"
                "• Check your internet connection")

else:
    # Welcome message
    st.info("👆 Configure your settings in the sidebar and click 'Start Analysis' to begin!")
    
    # Feature highlights
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)
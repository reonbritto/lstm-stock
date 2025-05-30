import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from model import train_lstm_model, predict_future_prices, calculate_metrics
import numpy as np
import ssl
import os

# Temporary SSL bypass (remove after fixing certificate issue)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Optional: Set CA bundle path (uncomment if using cacert.pem)
# os.environ["REQUESTS_CA_BUNDLE"] = "C:\\Users\\ReonBritto\\cacert.pem"

# Set page configuration for a modern look
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

# Sidebar for user inputs
st.sidebar.header("Stock Prediction Settings")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL,MSFT):", "AAPL").split(",")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
prediction_days = st.sidebar.selectbox("Prediction Horizon (Days)", [7, 15, 30], index=2)

# Main title and description
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("Analyze historical stock trends and predict future prices using an advanced LSTM model.")

# Fetch and display data when the button is clicked
if st.sidebar.button("Analyze & Predict"):
    st.subheader("Stock Analysis Results")
    
    # Initialize containers for charts and metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create subplot for historical and predicted prices
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=("Stock Price Trends", "Trading Volume"),
                            vertical_spacing=0.1)
        
        for ticker in [t.strip() for t in tickers]:
            # Fetch historical data
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            except Exception as e:
                st.error(f"Failed to fetch data for {ticker}: {str(e)}")
                continue
            
            if not stock_data.empty:
                # Display historical data
                st.write(f"### {ticker} Historical Data")
                st.dataframe(stock_data[['Close', 'Volume', '20_MA']].tail())

                # Train LSTM model and predict
                model, scaler, X_test, y_test = train_lstm_model(stock_data)
                future_predictions, future_dates, test_predictions, test_actual = predict_future_prices(
                    model, scaler, stock_data, days=prediction_days
                )

                # Plot historical and predicted prices
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                                       mode='lines', name=f"{ticker} Historical", line=dict(width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['20_MA'], 
                                       mode='lines', name=f"{ticker} 20-Day MA", line=dict(dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, 
                                       mode='lines', name=f"{ticker} Predicted", line=dict(dash='dash', width=2)), row=1, col=1)
                
                # Plot volume
                fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], 
                                   name=f"{ticker} Volume", opacity=0.5), row=2, col=1)
                
                # Calculate and display metrics
                mae, rmse = calculate_metrics(test_actual, test_predictions)
                with col2:
                    st.write(f"### {ticker} Prediction Metrics")
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
            else:
                st.error(f"No data available for {ticker}.")
        
        # Update chart layout
        fig.update_layout(height=600, title_text="Stock Price and Volume Analysis", 
                         xaxis_title="Date", yaxis_title="Price (USD)", 
                         xaxis2_title="Date", yaxis2_title="Volume")
        fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.write("**Note**: Predictions are based on historical closing prices, volume, and 20-day moving average. Past performance is not indicative of future results.")
else:
    st.info("Enter tickers and select a date range, then click 'Analyze & Predict' to view results.")
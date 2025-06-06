import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def apply_custom_css():
    """Apply professional custom CSS styling"""
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #1f77b4;
            --secondary-color: #ff7f0e;
            --success-color: #2ca02c;
            --danger-color: #d62728;
            --warning-color: #ff7f0e;
            --info-color: #17a2b8;
            --dark-color: #343a40;
            --light-color: #f8f9fa;
        }

        /* Header styling */
        .main-header {
            font-size: 3.5rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Professional card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: transform 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        /* Navigation styling */
        .nav-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }

        /* Button enhancements */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }

        /* Chart container */
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }

        /* Status indicators */
        .status-positive {
            color: var(--success-color);
            font-weight: bold;
        }

        .status-negative {
            color: var(--danger-color);
            font-weight: bold;
        }

        .status-neutral {
            color: var(--warning-color);
            font-weight: bold;
        }

        /* Performance metrics table */
        .performance-table {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 5px;
        }

        /* Loading animation */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 3rem;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2.5rem;
            }
            
            .metric-card {
                margin: 0.5rem 0;
                padding: 1rem;
            }
        }

        /* Animation keyframes */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }

        /* Professional table styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 0.5rem;
            border-radius: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 60px;
            border-radius: 10px;
            padding: 0 24px;
            background: white;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }

        /* Professional alerts */
        .custom-alert {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid;
        }

        .alert-success {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }

        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        .alert-danger {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }

        .alert-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
    </style>
    """, unsafe_allow_html=True)

def create_professional_header(title, subtitle):
    """Create a professional header with gradient styling"""
    st.markdown(f'<h1 class="main-header fade-in">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 3rem;" class="fade-in">
        <p style="font-size: 1.3rem; color: #6c757d; font-weight: 300;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_cards(metrics_data):
    """Create professional metric cards"""
    cols = st.columns(len(metrics_data))
    
    for i, (label, value, delta, color) in enumerate(metrics_data):
        with cols[i]:
            delta_html = f"<small style='color: {color};'>{delta}</small>" if delta else ""
            st.markdown(f"""
            <div class="metric-card fade-in">
                <h3 style="margin: 0; font-size: 1.1rem; opacity: 0.9;">{label}</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{value}</h2>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)

def create_enhanced_chart(data, title, chart_type="line"):
    """Create enhanced charts with professional styling"""
    fig = go.Figure()
    
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#667eea', width=3)
        ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_performance_summary(metrics):
    """Create a professional performance summary"""
    performance_color = "#28a745" if metrics['r2'] > 0.8 else "#ffc107" if metrics['r2'] > 0.6 else "#dc3545"
    
    st.markdown(f"""
    <div class="performance-table fade-in">
        <h3 style="text-align: center; color: #2c3e50; margin-bottom: 1rem;">
            📊 Model Performance Summary
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h4 style="color: #6c757d; margin: 0;">MAE</h4>
                <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 0.5rem 0;">${metrics['mae']:.2f}</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h4 style="color: #6c757d; margin: 0;">RMSE</h4>
                <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 0.5rem 0;">${metrics['rmse']:.2f}</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h4 style="color: #6c757d; margin: 0;">MAPE</h4>
                <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 0.5rem 0;">{metrics['mape']:.2f}%</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h4 style="color: #6c757d; margin: 0;">R² Score</h4>
                <p style="font-size: 1.5rem; font-weight: bold; color: {performance_color}; margin: 0.5rem 0;">{metrics['r2']:.3f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_trading_recommendation(price_change_pct, confidence_level):
    """Create professional trading recommendation"""
    if price_change_pct > 5:
        rec_color, rec_icon, rec_text = "#28a745", "🟢", "STRONG BUY"
        rec_desc = "Model predicts significant upward movement"
    elif price_change_pct > 2:
        rec_color, rec_icon, rec_text = "#17a2b8", "🔵", "BUY"
        rec_desc = "Model predicts moderate upward movement"
    elif price_change_pct > -2:
        rec_color, rec_icon, rec_text = "#ffc107", "🟡", "HOLD"
        rec_desc = "Model predicts sideways movement"
    elif price_change_pct > -5:
        rec_color, rec_icon, rec_text = "#fd7e14", "🟠", "SELL"
        rec_desc = "Model predicts moderate downward movement"
    else:
        rec_color, rec_icon, rec_text = "#dc3545", "🔴", "STRONG SELL"
        rec_desc = "Model predicts significant downward movement"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {rec_color}15 0%, {rec_color}05 100%);
        border: 2px solid {rec_color};
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    " class="fade-in">
        <h2 style="color: {rec_color}; margin: 0; font-size: 1.8rem;">
            {rec_icon} {rec_text}
        </h2>
        <p style="color: #6c757d; margin: 0.5rem 0; font-size: 1.1rem;">
            {rec_desc}
        </p>
        <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
            Confidence Level: <strong>{confidence_level}</strong> | 
            Price Change: <strong>{price_change_pct:+.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_footer():
    """Create professional footer"""
    st.markdown("""
    <div class="footer">
        <p>
            <strong>AI Stock Predictor</strong> - Advanced Machine Learning for Financial Markets<br>
            <small>⚠️ For educational and research purposes only. Not financial advice.</small>
        </p>
        <p style="font-size: 0.8rem; color: #adb5bd;">
            Powered by TensorFlow • LSTM Neural Networks • Technical Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_loading_animation(text="Processing..."):
    """Show professional loading animation"""
    st.markdown(f"""
    <div class="loading-container">
        <div style="text-align: center;">
            <div style="
                display: inline-block;
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            "></div>
            <p style="color: #6c757d; font-size: 1.1rem;">{text}</p>
        </div>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """, unsafe_allow_html=True)

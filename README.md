# AI Stock Price Predictor

This project is an interactive web application for stock price prediction using deep learning and machine learning models. It combines financial time-series data, technical indicators, and (optionally) news sentiment to forecast future stock prices.

## Features

- **Historical Data Fetching:** Retrieves stock price data using Yahoo Finance.
- **Feature Engineering:** Computes technical indicators (moving averages, RSI, momentum, etc.).
- **Sentiment Analysis (optional):** Placeholder for FinBERT/HuggingFace-based news sentiment integration.
- **Deep Learning Model:** LSTM-based neural network with attention and advanced architecture.
- **Tabular Model:** XGBoost regressor for tabular feature-based prediction.
- **Prediction Visualization:** Interactive charts for historical and predicted prices, volume, and confidence intervals.
- **Performance Metrics:** MAE, RMSE, MAPE, R².
- **Downloadable Results:** Export predictions as CSV.
- **Extensible:** Ready for integration with Hugging Face Transformers for advanced text-based triggers.
- **Simplified LSTM Pipeline:** Quick-to-train model using key indicators.
- **Advanced Technical Indicators:** EMA-50, EMA-200, MACD, Bollinger Bands, Stochastic Oscillator, Volume Change.
- **Improved Data Prep:** RobustScaler normalization, business-day future date generation.

## Project Structure

```
.
├── main.py                # Streamlit app UI and workflow
├── model.py               # Feature engineering, model training, prediction, evaluation
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## How It Works

1. **User selects a stock, date range, and prediction settings in the sidebar.**
2. **App fetches historical price data and computes technical indicators.**
3. **(Optional) News headlines can be merged for sentiment features.**
4. **LSTM or XGBoost model is trained on the features.**
5. **Future prices are predicted and visualized with performance metrics.**
6. **Results can be downloaded for further analysis.**

## Setup

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # (ensure tensorflow and sklearn versions compatible)
    ```

2. **Run the app:**
    ```bash
    streamlit run main.py
    ```

3. **(Optional) For Hugging Face Transformers features:**
    ```bash
    pip install transformers torch
    ```

## Notes

- The sentiment and transformer features are placeholders; integrate real models as needed.
- The app is for educational and research purposes only.
- You can switch between the full-featured model and the simplified pipeline by importing `train_and_evaluate` etc.
- Adjust `time_steps`, epochs, and batch size for your data and compute.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Credits

- Built with [Streamlit](https://streamlit.io/), [TensorFlow/Keras](https://www.tensorflow.org/), [XGBoost](https://xgboost.ai/), [yfinance](https://github.com/ranaroussi/yfinance), and [Hugging Face Transformers](https://huggingface.co/transformers/).

---

## Model Optimization Strategies

Improving the prediction accuracy of financial models is an ongoing process. Here are several strategies to potentially enhance the performance of this application:

### 1. Data Enhancement
-   **More Historical Data:** Using a longer time series for training can help the model learn more complex patterns.
-   **Higher Frequency Data:** Consider using intraday data (e.g., hourly or minute-by-minute) if your prediction horizon is short-term.
-   **Alternative Data Sources:**
    -   **Macroeconomic Indicators:** Incorporate data like interest rates, inflation, GDP growth, unemployment rates.
    -   **Market Indices:** Include data from relevant market indices (e.g., S&P 500, NASDAQ).
    -   **Social Media Trends:** Analyze sentiment from platforms like X (formerly Twitter) or Reddit using NLP.
    -   **Fundamental Data:** Integrate company-specific fundamentals like P/E ratio, earnings reports, debt-to-equity ratio.

### 2. Advanced Feature Engineering
-   **Sophisticated Technical Indicators:** Explore a wider range of indicators (e.g., MACD, Bollinger Bands, Ichimoku Cloud, Fibonacci retracements).
-   **Interaction Features:** Create features by combining existing ones (e.g., `Close * Volume`, `High - Low`).
-   **Feature Scaling:** Experiment with different scalers (e.g., `MinMaxScaler`, `StandardScaler`) beyond `RobustScaler`.
-   **Feature Selection:** Use techniques like SHAP, LIME, or scikit-learn's feature selection methods to identify and use only the most impactful features.
-   **Improved Sentiment Analysis:**
    -   Implement a dedicated financial sentiment model (e.g., FinBERT) instead of the placeholder.
    -   Use the loaded DeepSeek model for more nuanced sentiment or event detection from news.
    -   Consider aspect-based sentiment analysis.

### 3. Model Architecture & Hyperparameters
-   **Hyperparameter Tuning:**
    -   Re-introduce a systematic hyperparameter optimization library like Optuna.
    -   For LSTM: Tune units, layers, dropout rates, learning rate, batch size, optimizer.
    -   For XGBoost: Tune `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`.
-   **LSTM Architecture:**
    -   Experiment with different recurrent layers (GRU, BiLSTM).
    -   Adjust the number of LSTM layers and units.
    -   Explore different attention mechanisms or refine the existing `MultiHeadAttention`.
-   **Transformer Models:** For time-series, consider using dedicated Transformer architectures like TimeSformer or Informer.
-   **Regularization:** Apply L1/L2 regularization to dense layers or adjust dropout rates more finely to prevent overfitting.

### 4. Training & Evaluation
-   **Time-Series Cross-Validation:** Use scikit-learn's `TimeSeriesSplit` for more robust model evaluation and hyperparameter tuning, as standard k-fold cross-validation is not suitable for time-series data.
-   **Custom Loss Functions:** Explore loss functions that might be more appropriate for financial forecasting (e.g., penalizing under-prediction more than over-prediction, or vice-versa, depending on the strategy).
-   **Ensemble Methods:**
    -   Combine predictions from multiple models (e.g., LSTM and XGBoost).
    -   Techniques include weighted averaging, stacking, or blending.

### 5. Advanced Techniques
-   **Reinforcement Learning:** For trading strategy optimization, explore Deep Reinforcement Learning agents (e.g., using libraries like FinRL).
-   **Bayesian Optimization:** For hyperparameter tuning, especially with computationally expensive models.
-   **Stateful LSTMs:** If there's a need to maintain state across batches for very long sequences.

Implementing these strategies requires careful experimentation and validation to ensure genuine performance improvements.

## Simplified Model & Indicators
We offer a lightweight variant optimized for faster iteration:
1. **prepare_data_simple**  
   - Computes: Close, Return, Volatility, MA_20, EMA_20, EMA_50, EMA_200, MACD & signal, Bollinger Bands, Stochastic K/D, Volume Change.  
   - Scales with `RobustScaler` and builds sliding windows.

2. **build_simple_lstm_model**  
   - Two LSTM layers (128→64 units) + Dropout  
   - Dense head for single-step forecasting

3. **train_and_evaluate**  
   - 80/20 train/test split  
   - Early stopping, reporting RMSE, MAE, MAPE, R²

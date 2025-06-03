import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, MultiHeadAttention, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Layer
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def add_technical_indicators(df):
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def add_sentiment_features(df):
    # Placeholder for news sentiment scores (e.g., from FinBERT)
    # In practice, use an API or NLP model to get sentiment scores
    df['Sentiment'] = np.random.uniform(-1, 1, len(df))  # Dummy sentiment scores
    return df

def prepare_data(data, time_steps=60):
    for col in ['Close', 'Volume']:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if len(data) < time_steps + 20:
        raise ValueError(f"Insufficient data. Need at least {time_steps + 20} data points.")
    df = data.copy()
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Return'] = df['Close'].pct_change(periods=5)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df = add_technical_indicators(df)
    df = add_sentiment_features(df)  # Add sentiment features
    df = df.fillna(method='ffill').fillna(method='bfill')
    feature_columns = [
        'Close', 'Volume', '20_MA', 'Price_Change', 'Volume_MA', 'Return', 'Volatility',
        'Log_Return', 'Momentum', 'RSI', 'EMA_20', 'Sentiment'
    ]
    features = df[feature_columns].values
    mask = np.isfinite(features).all(axis=1)
    features = features[mask]
    df_clean = df[mask]
    return features, df_clean, feature_columns

def build_lstm_model(time_steps, n_features):
    # Fixed hyperparameters
    u1, u2, u3 = 256, 128, 64
    d1, d2, d3 = 0.3, 0.2, 0.1

    inputs = Input(shape=(time_steps, n_features))
    x = Conv1D(64, 3, activation='relu', padding='causal')(inputs)
    x = BatchNormalization()(x)

    x = LSTM(u1, return_sequences=True)(x)
    x = Dropout(d1)(x)
    x = BatchNormalization()(x)

    x = LSTM(u2, return_sequences=True)(x)
    x = Dropout(d2)(x)
    x = BatchNormalization()(x)

    x = LSTM(u3, return_sequences=True)(x)
    x = Dropout(d3)(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=u3)(x, x)
    x = BatchNormalization()(attn)
    x = Dropout(d3)(x)

    # Pool over time to get a vector
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mape'],
        run_eagerly=False
    )
    return model

def train_lstm_model(data, time_steps=60):
    try:
        if '20_MA' not in data.columns:
            data = data.copy()
            data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
        features, df_clean, feature_columns = prepare_data(data, time_steps)
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(features)
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        if len(X) == 0:
            raise ValueError("No sequences could be created. Check your data.")
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train the model
        model = build_lstm_model(time_steps, len(feature_columns))
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00005)
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        return model, scaler, X_test, y_test, df_clean, feature_columns, history
    except Exception as e:
        raise Exception(f"Error training model: {str(e)}")

def predict_future_prices(model, scaler, data, feature_columns, days=30, time_steps=60):
    try:
        if '20_MA' not in data.columns:
            data = data.copy()
            data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
        features, df_clean, _ = prepare_data(data, time_steps)
        scaled_data = scaler.transform(features)
        last_sequence = scaled_data[-time_steps:].copy()
        predictions = []
        current_sequence = last_sequence.reshape(1, time_steps, len(feature_columns))
        center = scaler.center_[0]
        scale = scaler.scale_[0]
        last_close = data['Close'].iloc[-1]
        for day in range(days):
            pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            pred = pred_scaled * scale + center
            # Clamp prediction to avoid negative/absurd values
            if np.isnan(pred) or np.isinf(pred) or pred < 0 or pred > last_close * 10:
                pred = last_close
            predictions.append(pred)
            last_row = current_sequence[0, -1].copy()
            last_row[0] = (pred - center) / scale
            if len(predictions) > 1 and predictions[-2] != 0:
                price_change = (pred - predictions[-2]) / predictions[-2]
            else:
                price_change = 0
            last_row[3] = price_change
            last_row[1] = last_row[1] * (1 + np.random.normal(0, 0.02))
            last_row[2] = last_row[2] * (1 + np.random.normal(0, 0.01))
            last_row[4] = last_row[1]
            last_row[5] = last_row[3]
            last_row[6] = np.std([seq[0] for seq in current_sequence[0]])
            last_row[7] = np.log(pred / last_close + 1e-9) if last_close != 0 else 0
            last_row[8] = pred - last_close
            last_row[9] = 50
            last_row[10] = pred
            last_row[11] = np.random.uniform(-1, 1)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = last_row
            last_close = pred
        # predictions are already in original scale
        last_date = data.index[-1]
        future_dates = []
        current_date = last_date
        for _ in range(days):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            future_dates.append(current_date)
        return np.array(predictions), future_dates
    except Exception as e:
        raise Exception(f"Error predicting future prices: {str(e)}")

def evaluate_model(model, scaler, X_test, y_test, feature_columns):
    """Evaluate model performance and return metrics as a dict"""
    try:
        # Predict and inverse-transform only the target (first) feature
        test_pred_scaled = model.predict(X_test, verbose=0).flatten()
        n = min(len(test_pred_scaled), len(y_test))
        test_pred_scaled = test_pred_scaled[-n:]
        actual_scaled = y_test[-n:]
        center = scaler.center_[0]
        scale = scaler.scale_[0]
        test_pred_inverse = test_pred_scaled * scale + center
        test_actual_inverse = actual_scaled * scale + center

        mae = float(mean_absolute_error(test_actual_inverse, test_pred_inverse))
        rmse = float(np.sqrt(mean_squared_error(test_actual_inverse, test_pred_inverse)))
        mape = float(np.mean(np.abs((test_actual_inverse - test_pred_inverse) / test_actual_inverse)) * 100)
        r2 = float(r2_score(test_actual_inverse, test_pred_inverse))
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "test_pred_inverse": test_pred_inverse,
            "test_actual_inverse": test_actual_inverse
        }
        return metrics
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def extract_sentiment_from_texts(texts):
    """
    Placeholder for FinBERT/HuggingFace sentiment extraction.
    Replace this with actual FinBERT inference or HuggingFace pipeline.
    """
    # Example: return np.random.uniform(-1, 1, len(texts))
    # For real use: load FinBERT or use transformers pipeline for sentiment
    return np.random.uniform(-1, 1, len(texts))

def merge_price_and_sentiment(price_df, news_df):
    """
    Merge price/indicator features with sentiment features.
    price_df: DataFrame with price and indicators (indexed by date)
    news_df: DataFrame with columns ['date', 'headline']
    Returns: merged DataFrame with sentiment score per day
    """
    # Extract sentiment for each news headline
    news_df['sentiment'] = extract_sentiment_from_texts(news_df['headline'])
    # Aggregate sentiment per day (mean)
    daily_sentiment = news_df.groupby('date')['sentiment'].mean()
    # Merge with price_df
    merged = price_df.copy()
    merged['date'] = merged.index.date
    merged = merged.merge(daily_sentiment, left_on='date', right_index=True, how='left')
    merged['sentiment'] = merged['sentiment'].fillna(0)
    merged = merged.drop(columns=['date'])
    return merged

def train_xgboost_model(df, feature_columns, target_column='Close'):
    """
    Train an XGBoost regressor on tabular features.
    Returns: trained model, feature importance
    """
    X = df[feature_columns].values
    y = df[target_column].values
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
    model.fit(X, y)
    importance = model.feature_importances_
    return model, importance

# Hugging Face Transformers: Load DeepSeek model for text-based triggers or embeddings
def load_deepseek_model():
    """
    Load the DeepSeek model and tokenizer from Hugging Face Hub.
    Returns: tokenizer, model
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
        return tokenizer, model
    except ImportError as e:
        raise ImportError("transformers and torch must be installed to use DeepSeek model. "
                          "Install with: pip install transformers torch") from e

# Example usage (call this only once and reuse the model/tokenizer as needed):
# tokenizer, model = load_deepseek_model()
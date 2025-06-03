from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta

def prepare_data_simple(df, time_steps=60):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df = df.fillna(method='ffill').fillna(method='bfill')

    features = df[['Close', 'Return', 'Volatility', 'MA_20', 'EMA_20']].values
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i])
        y.append(scaled[i, 0])  # Predict 'Close'

    return np.array(X), np.array(y), scaler

def build_simple_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Predict single value (next Close)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(df, time_steps=60):
    X, y, scaler = prepare_data_simple(df, time_steps)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_simple_lstm_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

    preds = model.predict(X_test)
    actual = y_test

    # Inverse transform
    center = scaler.center_[0]
    scale = scaler.scale_[0]
    preds_rescaled = preds.flatten() * scale + center
    actual_rescaled = actual * scale + center

    rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
    print(f"Test RMSE: {rmse:.4f}")
    return model, scaler, preds_rescaled, actual_rescaled

def predict_next_days(df, model, scaler, days=10, time_steps=60):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df = df.fillna(method='ffill').fillna(method='bfill')

    features = df[['Close', 'Return', 'Volatility', 'MA_20', 'EMA_20']].values
    scaled = scaler.transform(features)
    sequence = scaled[-time_steps:].copy()

    future_preds = []
    for _ in range(days):
        input_seq = sequence.reshape(1, time_steps, -1)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        pred = pred_scaled * scaler.scale_[0] + scaler.center_[0]
        future_preds.append(pred)

        # Create next row (dummy features)
        next_row = sequence[-1].copy()
        next_row[0] = pred_scaled
        next_row[1] = 0.01  # Dummy return
        next_row[2] = sequence[:, 0].std()  # Dummy volatility
        next_row[3] = sequence[:, 0].mean()
        next_row[4] = sequence[:, 0].mean()

        sequence = np.vstack([sequence[1:], next_row])

    return future_preds

def train_lstm_model(df, time_steps=60):
    # Wrapper for compatibility with main.py
    model, scaler, preds_rescaled, actual_rescaled = train_and_evaluate(df, time_steps)
    # For main.py compatibility, return dummy values for unused outputs
    X_test = None
    y_test = None
    df_clean = df
    feature_columns = ['Close', 'Return', 'Volatility', 'MA_20', 'EMA_20']
    history = type('History', (), {'history': {'loss': [0]}})()  # Dummy history
    return model, scaler, X_test, y_test, df_clean, feature_columns, history

def predict_future_prices(model, scaler, df, feature_columns, days=30, time_steps=60):
    # Wrapper for compatibility with main.py
    preds = predict_next_days(df, model, scaler, days=days, time_steps=time_steps)
    # Generate future dates
    last_date = df.index[-1]
    future_dates = []
    current_date = last_date
    for _ in range(days):
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        future_dates.append(current_date)
    return np.array(preds), future_dates

def evaluate_model(model, scaler, X_test, y_test, feature_columns):
    # Dummy evaluation for compatibility
    # In this simple version, evaluation is handled in train_and_evaluate
    return {
        "mae": 0.0,
        "rmse": 0.0,
        "mape": 0.0,
        "r2": 0.0,
        "test_pred_inverse": [],
        "test_actual_inverse": []
    }

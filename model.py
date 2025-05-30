import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_lstm_model(data, time_steps=60):
    # Calculate 20-day moving average
    data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
    
    # Prepare features: Close, Volume, 20_MA
    features = data[['Close', 'Volume', '20_MA']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])  # Predict Close price
    
    X, y = np.array(X), np.array(y)
    
    # Split data for test set (last 20% for evaluation)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 3)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    return model, scaler, X_test, y_test

def predict_future_prices(model, scaler, data, days=30, time_steps=60):
    # Calculate 20-day moving average
    data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
    
    # Prepare features
    features = data[['Close', 'Volume', '20_MA']].values
    scaled_data = scaler.transform(features)
    
    # Get the last sequence for prediction
    last_sequence = scaled_data[-time_steps:]
    last_sequence = np.reshape(last_sequence, (1, time_steps, 3))
    
    # Predict future prices
    predictions = []
    for _ in range(days):
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0, 0])
        # Simulate volume and MA for future steps (use last known values)
        last_volume = last_sequence[0, -1, 1]
        last_ma = last_sequence[0, -1, 2]
        new_row = np.array([[pred[0, 0], last_volume, last_ma]])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1] = new_row
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    dummy_features = np.zeros((len(predictions), 2))  # Dummy volume and MA for inverse transform
    predictions_full = np.hstack([predictions, dummy_features])
    predictions = scaler.inverse_transform(predictions_full)[:, 0]
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=x) for x in range(1, days+1)]
    
    # Evaluate model on test set
    test_predictions = model.predict(X_test, verbose=0).flatten()
    test_actual = y_test
    test_predictions = scaler.inverse_transform(
        np.hstack([test_predictions.reshape(-1, 1), np.zeros((len(test_predictions), 2))])
    )[:, 0]
    test_actual = scaler.inverse_transform(
        np.hstack([test_actual.reshape(-1, 1), np.zeros((len(test_actual), 2))])
    )[:, 0]
    
    return predictions, future_dates, test_predictions, test_actual

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse
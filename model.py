import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def prepare_data(data, time_steps=60):
    """Prepare and clean data for LSTM model"""
    # Ensure we have enough data
    if len(data) < time_steps + 20:
        raise ValueError(f"Insufficient data. Need at least {time_steps + 20} data points.")
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Calculate technical indicators
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    
    # Forward fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Select features
    feature_columns = ['Close', 'Volume', '20_MA', 'Price_Change']
    features = df[feature_columns].values
    
    # Remove any rows with NaN or infinite values
    mask = np.isfinite(features).all(axis=1)
    features = features[mask]
    df_clean = df[mask]
    
    return features, df_clean, feature_columns

def train_lstm_model(data, time_steps=60):
    """Train LSTM model with improved architecture and validation"""
    try:
        # Ensure 20_MA is present
        if '20_MA' not in data.columns:
            data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
        
        # Prepare data
        features, df_clean, feature_columns = prepare_data(data, time_steps)
        
        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i])
            y.append(scaled_data[i, 0])  # Predict Close price (first feature)
        
        X, y = np.array(X), np.array(y)
        
        if len(X) == 0:
            raise ValueError("No sequences could be created. Check your data.")
        
        # Split data (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build improved LSTM model
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(time_steps, len(feature_columns))),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units=80, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units=60),
            Dropout(0.2),
            
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        
        # Callbacks for better training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return model, scaler, X_test, y_test, df_clean, feature_columns, history
        
    except Exception as e:
        raise Exception(f"Error training model: {str(e)}")

def predict_future_prices(model, scaler, data, days=30, time_steps=60):
    """Predict future prices with improved logic"""
    try:
        # Ensure 20_MA is present
        if '20_MA' not in data.columns:
            data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
        
        # Prepare data
        features, df_clean, _ = prepare_data(data, time_steps)
        
        # Get the last sequence for prediction
        scaled_data = scaler.transform(features)
        last_sequence = scaled_data[-time_steps:].copy()
        
        # Predict future prices
        predictions = []
        current_sequence = last_sequence.reshape(1, time_steps, len(feature_columns))
        
        for day in range(days):
            # Predict next price
            pred = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Create new row for next prediction
            # Use last known values for other features and update them gradually
            last_row = current_sequence[0, -1].copy()
            
            # Update price
            last_row[0] = pred
            
            # Simulate price change
            if len(predictions) > 1:
                price_change = (pred - predictions[-2]) / predictions[-2] if predictions[-2] != 0 else 0
                last_row[3] = price_change
            
            # Keep volume and MA relatively stable with small random variations
            last_row[1] = last_row[1] * (1 + np.random.normal(0, 0.02))  # Volume with 2% noise
            last_row[2] = last_row[2] * (1 + np.random.normal(0, 0.01))  # MA with 1% noise
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = last_row
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_features = np.tile(scaled_data[-1, 1:], (len(predictions), 1))
        predictions_full = np.hstack([predictions, dummy_features])
        predictions_inverse = scaler.inverse_transform(predictions_full)[:, 0]
        
        # Generate future dates (excluding weekends for trading days)
        last_date = data.index[-1]
        future_dates = []
        current_date = last_date
        
        for _ in range(days):
            current_date += timedelta(days=1)
            # Skip weekends (Saturday=5, Sunday=6)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            future_dates.append(current_date)
        
        return predictions_inverse, future_dates
        
    except Exception as e:
        raise Exception(f"Error predicting future prices: {str(e)}")

def evaluate_model(model, scaler, X_test, y_test, feature_columns):
    """Evaluate model performance"""
    try:
        # Make predictions on test set
        test_predictions = model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform both predictions and actual values
        dummy_features = np.zeros((len(test_predictions), len(feature_columns) - 1))
        
        test_pred_full = np.hstack([test_predictions.reshape(-1, 1), dummy_features])
        test_actual_full = np.hstack([y_test.reshape(-1, 1), dummy_features])
        
        test_pred_inverse = scaler.inverse_transform(test_pred_full)[:, 0]
        test_actual_inverse = scaler.inverse_transform(test_actual_full)[:, 0]
        
        # Calculate metrics
        mae = mean_absolute_error(test_actual_inverse, test_pred_inverse)
        rmse = np.sqrt(mean_squared_error(test_actual_inverse, test_pred_inverse))
        mape = np.mean(np.abs((test_actual_inverse - test_pred_inverse) / test_actual_inverse)) * 100
        
        return mae, rmse, mape, test_pred_inverse, test_actual_inverse
        
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def calculate_metrics(actual, predicted):
    """Calculate various performance metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape
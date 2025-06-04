import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

def prepare_data_simple(df, time_steps=60, market='US'):
    df = df.copy()
    # Add standard features
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    # Add Indian market-specific features if available
    if market == 'India' and 'FII_Net' in df.columns:
        df['FII_Net'] = df['FII_Net']  # Foreign Institutional Investor net flows
    if market == 'India' and 'Sector_Index' in df.columns:
        df['Sector_Index'] = df['Sector_Index']  # Sector-specific index
    
    df = df.fillna(method='ffill').fillna(method='bfill')

    features = ['Close', 'Return', 'Volatility', 'MA_20', 'EMA_20', 'RSI', 'BB_High', 'BB_Low']
    if market == 'India':
        if 'FII_Net' in df.columns:
            features.append('FII_Net')
        if 'Sector_Index' in df.columns:
            features.append('Sector_Index')
    
    feature_data = df[features].values
    scaler = RobustScaler()
    scaled = scaler.fit_transform(feature_data)

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i])
        y.append(scaled[i, 0])  # Predict 'Close'
    return np.array(X), np.array(y), scaler, features

def build_simple_lstm_model(input_shape, units=64, dropout=0.3):
    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=False), input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(df, time_steps=60, n_splits=5, market='US', hyperparams=None):
    X, y, scaler, feature_columns = prepare_data_simple(df, time_steps, market)
    
    # Default hyperparameters
    units = hyperparams.get('units', 64) if hyperparams else 64
    dropout = hyperparams.get('dropout', 0.3) if hyperparams else 0.3
    batch_size = hyperparams.get('batch_size', 32) if hyperparams else 32
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_simple_lstm_model((X.shape[1], X.shape[2]), units, dropout)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        model.fit(X_train, y_train, epochs=50, batch_size=batch_size, 
                  validation_data=(X_test, y_test), callbacks=[early_stop, lr_scheduler], verbose=0)

        preds = model.predict(X_test, verbose=0).flatten()
        actual = y_test
        center = scaler.center_[0]
        scale = scaler.scale_[0]
        preds_rescaled = preds * scale + center
        actual_rescaled = actual * scale + center
        rmses.append(np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled)))

    print(f"{market} Market Cross-Validated RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # Train final model
    model = build_simple_lstm_model((X.shape[1], X.shape[2]), units, dropout)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, 
                        validation_data=(X_test, y_test), callbacks=[early_stop, lr_scheduler], verbose=0)

    preds = model.predict(X_test, verbose=0)
    actual = y_test
    center = scaler.center_[0]
    scale = scaler.scale_[0]
    preds_rescaled = preds.flatten() * scale + center
    actual_rescaled = actual * scale + center

    rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
    print(f"{market} Market Test RMSE: {rmse:.4f}")
    return model, scaler, preds_rescaled, actual_rescaled, history, X_test, y_test, feature_columns

def predict_next_days(df, model, scaler, days=10, time_steps=60, market='US'):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    if market == 'India' and 'FII_Net' in df.columns:
        df['FII_Net'] = df['FII_Net']
    if market == 'India' and 'Sector_Index' in df.columns:
        df['Sector_Index'] = df['Sector_Index']
    
    df = df.fillna(method='ffill').fillna(method='bfill')

    features = ['Close', 'Return', 'Volatility', 'MA_20', 'EMA_20', 'RSI', 'BB_High', 'BB_Low']
    if market == 'India':
        if 'FII_Net' in df.columns:
            features.append('FII_Net')
        if 'Sector_Index' in df.columns:
            features.append('Sector_Index')
    
    feature_data = df[features].values
    scaled = scaler.transform(feature_data)
    sequence = scaled[-time_steps:].copy()

    future_preds = []
    for _ in range(days):
        input_seq = sequence.reshape(1, time_steps, -1)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        pred = pred_scaled * scaler.scale_[0] + scaler.center_[0]
        future_preds.append(pred)

        next_row = sequence[-1].copy()
        next_row[0] = pred_scaled
        next_row[1] = sequence[-10:, 1].mean()
        next_row[2] = sequence[-10:, 2].mean()
        next_row[3] = np.mean(sequence[-20:, 0])
        next_row[4] = np.mean(sequence[-20:, 0])
        next_row[5] = sequence[-14:, 5].mean()
        next_row[6] = sequence[-20:, 6].mean()
        next_row[7] = sequence[-20:, 7].mean()
        if market == 'India' and len(features) > 8:
            if 'FII_Net' in features:
                next_row[8] = sequence[-10:, 8].mean() if len(sequence[0]) > 8 else 0
            if 'Sector_Index' in features:
                next_row[9 if 'FII_Net' in features else 8] = sequence[-10:, 9 if 'FII_Net' in features else 8].mean() if len(sequence[0]) > 8 else 0
        sequence = np.vstack([sequence[1:], next_row])

    return future_preds

def predict_future_prices(model, scaler, df, feature_columns, days=30, time_steps=60, market='US'):
    preds = predict_next_days(df, model, scaler, days=days, time_steps=time_steps, market=market)
    last_date = df.index[-1]
    future_dates = []
    current_date = last_date
    for _ in range(days):
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:  # Skip weekends
            current_date += timedelta(days=1)
        future_dates.append(current_date)
    return np.array(preds), future_dates

def evaluate_model(model, scaler, X_test, y_test, feature_columns):
    try:
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
        mape = float(np.mean(np.abs((test_actual_inverse - test_pred_inverse) / 
                                    (test_actual_inverse + 1e-10))) * 100)
        r2 = float(r2_score(test_actual_inverse, test_pred_inverse))
        
        direction_pred = np.sign(test_pred_inverse[1:] - test_pred_inverse[:-1])
        direction_actual = np.sign(test_actual_inverse[1:] - test_actual_inverse[:-1])
        directional_accuracy = float(np.mean(direction_pred == direction_actual) * 100)

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "test_pred_inverse": test_pred_inverse,
            "test_actual_inverse": test_actual_inverse
        }
        return metrics
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def train_lstm_model(df, time_steps=60, market='US', hyperparams=None):
    model, scaler, preds_rescaled, actual_rescaled, history, X_test, y_test, feature_columns = train_and_evaluate(df, time_steps, market=market, hyperparams=hyperparams)
    df_clean = df
    return model, scaler, X_test, y_test, df_clean, feature_columns, history

def train_ensemble_model(df, time_steps=60, market='US'):
    X, y, scaler, feature_columns = prepare_data_simple(df, time_steps, market)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for tree-based models
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    lstm_model = build_simple_lstm_model((X.shape[1], X.shape[2]))
    rf_model = RandomForestRegressor(n_estimators=100)
    xgb_model = XGBRegressor(n_estimators=100)

    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    rf_model.fit(X_train_2d, y_train)
    xgb_model.fit(X_train_2d, y_train)

    ensemble = VotingRegressor(estimators=[
        ('lstm', lstm_model), ('rf', rf_model), ('xgb', xgb_model)
    ])
    ensemble.fit(X_train_2d, y_train)

    preds = ensemble.predict(X_test_2d)
    center = scaler.center_[0]
    scale = scaler.scale_[0]
    preds_rescaled = preds * scale + center
    actual_rescaled = y_test * scale + center

    rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
    print(f"{market} Market Ensemble Test RMSE: {rmse:.4f}")
    return ensemble, scaler, preds_rescaled, actual_rescaled, None, X_test, y_test, feature_columns

def plot_predictions(actual, predicted, dates=None):
    if dates is None:
        dates = range(len(actual))
    
    chart_data = {
        "type": "line",
        "data": {
            "labels": [str(d) for d in dates] if isinstance(dates, list) else list(dates),
            "datasets": [
                {
                    "label": "Actual Prices",
                    "data": actual.tolist(),
                    "borderColor": "#1f77b4",
                    "fill": False
                },
                {
                    "label": "Predicted Prices",
                    "data": predicted.tolist(),
                    "borderColor": "#ff7f0e",
                    "fill": False
                }
            ]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "Date"}},
                "y": {"title": {"display": True, "text": "Price"}}
            }
        }
    }
    return chart_data

# Example usage:
# df_us = pd.read_csv('us_stock_data.csv', index_col='Date', parse_dates=True)
# df_india = pd.read_csv('india_stock_data.csv', index_col='Date', parse_dates=True)
# hyperparams_india = {'units': 128, 'dropout': 0.4, 'batch_size': 16}
# model_us, scaler_us, X_test_us, y_test_us, df_clean_us, feature_columns_us, history_us = train_lstm_model(df_us, market='US')
# model_india, scaler_india, X_test_india, y_test_india, df_clean_india, feature_columns_india, history_india = train_lstm_model(df_india, market='India', hyperparams=hyperparams_india)
# metrics_us = evaluate_model(model_us, scaler_us, X_test_us, y_test_us, feature_columns_us)
# metrics_india = evaluate_model(model_india, scaler_india, X_test_india, y_test_india, feature_columns_india)
# print("US Metrics:", metrics_us)
# print("India Metrics:", metrics_india)
# chart_us = plot_predictions(metrics_us['test_actual_inverse'], metrics_us['test_pred_inverse'], df_clean_us.index[-len(X_test_us):])
# chart_india = plot_predictions(metrics_india['test_actual_inverse'], metrics_india['test_pred_inverse'], df_clean_india.index[-len(X_test_india):])
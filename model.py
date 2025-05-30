import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    df = df.fillna(method='ffill').fillna(method='bfill')
    feature_columns = [
        'Close', 'Volume', '20_MA', 'Price_Change', 'Volume_MA', 'Return', 'Volatility',
        'Log_Return', 'Momentum', 'RSI', 'EMA_20'
    ]
    features = df[feature_columns].values
    mask = np.isfinite(features).all(axis=1)
    features = features[mask]
    df_clean = df[mask]
    return features, df_clean, feature_columns

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
        # Functional API for LSTM with Attention
        inputs = Input(shape=(time_steps, len(feature_columns)))
        x = LSTM(256, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Attention()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
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
        for day in range(days):
            pred = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            last_row = current_sequence[0, -1].copy()
            last_row[0] = pred
            if len(predictions) > 1:
                price_change = (pred - predictions[-2]) / predictions[-2] if predictions[-2] != 0 else 0
                last_row[3] = price_change
            last_row[1] = last_row[1] * (1 + np.random.normal(0, 0.02))
            last_row[2] = last_row[2] * (1 + np.random.normal(0, 0.01))
            last_row[4] = last_row[1]
            last_row[5] = last_row[3]
            last_row[6] = np.std([seq[0] for seq in current_sequence[0]])
            last_row[7] = np.log(pred / current_sequence[0, -1, 0] + 1e-9)
            last_row[8] = pred - current_sequence[0, -10, 0] if time_steps > 10 else 0
            last_row[9] = 50
            last_row[10] = pred
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = last_row
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_features = np.tile(scaled_data[-1, 1:], (len(predictions), 1))
        predictions_full = np.hstack([predictions, dummy_features])
        predictions_inverse = scaler.inverse_transform(predictions_full)[:, 0]
        last_date = data.index[-1]
        future_dates = []
        current_date = last_date
        for _ in range(days):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            future_dates.append(current_date)
        return predictions_inverse, future_dates
    except Exception as e:
        raise Exception(f"Error predicting future prices: {str(e)}")

def evaluate_model(model, scaler, X_test, y_test, feature_columns):
    try:
        test_predictions = model.predict(X_test, verbose=0).flatten()
        dummy_features = np.zeros((len(test_predictions), len(feature_columns) - 1))
        test_pred_full = np.hstack([test_predictions.reshape(-1, 1), dummy_features])
        test_actual_full = np.hstack([y_test.reshape(-1, 1), dummy_features])
        test_pred_inverse = scaler.inverse_transform(test_pred_full)[:, 0]
        test_actual_inverse = scaler.inverse_transform(test_actual_full)[:, 0]
        mae = mean_absolute_error(test_actual_inverse, test_pred_inverse)
        rmse = np.sqrt(mean_squared_error(test_actual_inverse, test_pred_inverse))
        mape = np.mean(np.abs((test_actual_inverse - test_pred_inverse) / test_actual_inverse)) * 100
        r2 = r2_score(test_actual_inverse, test_pred_inverse)
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
import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def get_stock_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    return df[['Close']].dropna()


def create_dataset(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


def evaluate_model(y_true, y_pred):
    metrics = {}

    # Root Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = math.sqrt(mse)

    # Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)

    # Mean Absolute Percentage Error
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    metrics['MAPE'] = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100

    # Coefficient of determination
    metrics['R2'] = r2_score(y_true, y_pred)

    # Directional Accuracy: percentage of correct direction predictions
    actual_directions = np.sign(np.diff(y_true.reshape(-1)))
    predicted_directions = np.sign(np.diff(y_pred.reshape(-1)))

    # Remove zero direction predictions (no change)
    valid_indices = np.where(actual_directions != 0)[0]
    if len(valid_indices) > 0:
        correct_directions = np.sum(actual_directions[valid_indices] == predicted_directions[valid_indices])
        metrics['DA'] = correct_directions / len(valid_indices) * 100
    else:
        metrics['DA'] = np.nan

    return metrics


def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    model.fit(X_train, y_train)
    return model


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_lstm_model(X_train, y_train):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    y_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_train.shape[1], 1)

    model = build_lstm_model((X_train.shape[1], 1))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_reshaped,
        y_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )

    class LSTMWrapper:
        def __init__(self, lstm_model, scaler):
            self.model = lstm_model
            self.scaler = scaler

        def predict(self, X):
            X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_train.shape[1], 1)
            y_pred_scaled = self.model.predict(X_reshaped, verbose=0)
            return self.scaler.inverse_transform(y_pred_scaled)

    return LSTMWrapper(model, scaler)


def compare_models(ticker, time_steps=10, test_size=0.2):
    df = get_stock_data(ticker)
    prices = df['Close'].values.reshape(-1, 1)

    X, y = create_dataset(prices, time_steps)
    X_flat = X.reshape(X.shape[0], X.shape[1])

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")

    models = {
        'XGBoost': train_xgboost_model(X_train, y_train.ravel()),
        'LSTM': train_lstm_model(X_train, y_train)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results[name] = metrics
        predictions[name] = y_pred

        print(f"Performance metrics for {name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison Summary:")
    print(comparison_df)

    return comparison_df, models


if __name__ == "__main__":
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    results_df, trained_models = compare_models("AAPL", time_steps=15)
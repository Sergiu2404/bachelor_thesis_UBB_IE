import random

import numpy as np
import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta
import os

class LSTMStockPricePredictor:
    def __init__(self, seq_length=60, model_path='E:\\saved_models\\lstm_prediction_model\\lstm_model.h5'):
        self.seq_length = seq_length
        self.model_path = model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def create_sequences(self, data):
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            X.append(data[i-self.seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def prepare_stock_data(
            self,
            ticker,
            train_start='2010-01-01',
            train_end='2020-12-31',
            val_start='2021-01-01',
            val_end='2023-01-01'
    ):
        data = yf.download(ticker, start=train_start, end=val_end)

        training_data = data['Close'][train_start:train_end]
        validation_data = data['Close'][val_start:val_end]

        training_set = training_data.values.reshape(-1, 1)
        validation_set = validation_data.values.reshape(-1, 1)

        training_set_scaled = self.scaler.fit_transform(training_set)
        validation_set_scaled = self.scaler.transform(validation_set)

        X_train, y_train = self.create_sequences(training_set_scaled)
        X_validation, y_validation = self.create_sequences(validation_set_scaled)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))

        return X_train, y_train, X_validation, y_validation, data

    def build_model(self, input_shape):
        model = models.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=32):
        if os.path.exists(self.model_path):
            print("load existing model")
            self.model = models.load_model(self.model_path)
        else:
            print("build, train, save new model")
            self.model = self.build_model((X_train.shape[1], 1))
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation), verbose=1)
            self.model.save(self.model_path)

    def prepare_latest_data(self, ticker):
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.seq_length * 2)).strftime('%Y-%m-%d')
        latest_data = yf.download(ticker, start=start_date, end=today)

        latest_prices = latest_data['Close'].values.reshape(-1, 1)
        latest_prices_scaled = self.scaler.transform(latest_prices)

        X_latest = np.array([latest_prices_scaled[-self.seq_length:, 0]])
        X_latest = X_latest.reshape((X_latest.shape[0], X_latest.shape[1], 1))

        return X_latest, latest_data

    def predict_next_12_months(self, X_latest):
        current_sequence = X_latest[0]
        predictions = []
        current_date = datetime.now()
        future_dates = []

        for i in range(12):
            future_date = (current_date + timedelta(days=30 * (i + 1)))
            future_dates.append(future_date)

            current_batch = current_sequence[-self.seq_length:].reshape(1, self.seq_length, 1)
            next_pred = self.model.predict(current_batch, verbose=0)
            predictions.append(next_pred[0, 0])

            influence = 0.7 + 0.03 * i  # increase influence of the last prediction
            weighted_pred = (current_sequence[-1] * influence + next_pred[0, 0]) / (1 + influence)
            current_sequence = np.append(current_sequence, weighted_pred)

        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array)

        return predictions_rescaled, future_dates

    def print_monthly_predictions(self, predictions, future_dates, ticker):
        print(f"\n{ticker} Monthly Price Predictions:")
        print(f"{'Month':<15} {'Predicted Price':<15}")
        print("-" * 40)

        for i in range(len(predictions)):
            month_str = future_dates[i].strftime('%b %Y')
            price_str = f"${predictions[i][0]:.2f}"
            print(f"{month_str:<15} {price_str:<15}")

    def run(self, ticker):
        print(f"Loading and preparing {ticker} data...")
        X_train, y_train, X_validation, y_validation, historical_data = self.prepare_stock_data(ticker)
        self.train_model(X_train, y_train, X_validation, y_validation)

        print(f"prepare {ticker} data for pred")
        latest_sequence, latest_data = self.prepare_latest_data(ticker)

        print("predict prices for the next 12 months")
        predictions, future_dates = self.predict_next_12_months(latest_sequence)
        self.print_monthly_predictions(predictions, future_dates, ticker)

# predictor = LSTMStockPricePredictor()
# predictor.run('SPY')
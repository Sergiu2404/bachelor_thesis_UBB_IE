import numpy as np
import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class StockPricePredictor:
    def __init__(self, look_back=60):
        """
        Initializes the StockPricePredictor with a specified look-back period.
        :param look_back: Number of previous days used for prediction.
        """
        self.look_back = look_back
        self.model = self.create_lstm_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL').
        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format.
        :return: DataFrame with stock price data.
        """
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df

    def prepare_data(self, data):
        """
        Prepares stock price data for LSTM model training.
        :param data: DataFrame containing stock price data.
        :return: Tuple (X, y, scaler) where X is input sequences, y is target values.
        """
        prices = data['Close'].values.reshape(-1, 1)
        prices_scaled = self.scaler.fit_transform(prices)

        X, y = [], []
        for i in range(self.look_back, len(prices_scaled)):
            X.append(prices_scaled[i - self.look_back:i, 0])
            y.append(prices_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def create_lstm_model(self):
        """
        Creates and compiles an LSTM model for stock price prediction.
        :return: Compiled LSTM model.
        """
        model = models.Sequential([
            layers.LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)),
            layers.Dropout(0.2),
            layers.LSTM(units=50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_future(self, last_sequence, num_days):
        """
        Predict future stock prices based on trained LSTM model.
        :param last_sequence: The last known sequence of stock prices.
        :param num_days: Number of days to predict.
        :return: List of predicted stock prices for the next num_days.
        """
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(num_days):
            current_sequence_reshaped = current_sequence.reshape(1, -1, 1)
            next_day_scaled = self.model.predict(current_sequence_reshaped)

            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_day_scaled

            future_predictions.append(next_day_scaled[0])

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        return future_predictions.flatten()

    def run_model(self, ticker, start_date, end_date, num_days_to_predict, epochs=50, batch_size=32):
        """
        Executes the complete stock price prediction workflow.
        :param ticker: Stock ticker symbol.
        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format.
        :param num_days_to_predict: Number of days to predict into the future.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :return: List of tuples (date, predicted_price).
        """
        print(f"Fetching data for {ticker}...")
        df = self.fetch_stock_data(ticker, start_date, end_date)
        X, y = self.prepare_data(df)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print("Training LSTM model...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

        last_sequence = X[-1]
        future_predictions = self.predict_future(last_sequence, num_days_to_predict)

        last_date = datetime.strptime(end_date, '%Y-%m-%d')
        future_dates = [last_date + timedelta(days=x + 1) for x in range(num_days_to_predict)]

        predictions = [(date.date(), price) for date, price in zip(future_dates, future_predictions)]

        print("\nPredicted prices for the next", num_days_to_predict, "days:")
        for date, price in predictions:
            print(f"{date}: ${price:.2f}")

        return predictions


# Testing the model
# if __name__ == "__main__":
#     predictor = StockPricePredictor()
#     predictions = predictor.run_model('AAPL', '2024-01-01', '2024-05-05', 10)
#     print("Test Predictions:", predictions)

# import numpy as np
# import pandas as pd
# import yfinance as yf

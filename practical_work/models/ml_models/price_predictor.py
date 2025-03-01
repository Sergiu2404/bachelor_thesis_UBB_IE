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

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def prepare_stock_data(ticker, train_start='2010-01-01', train_end='2020-12-31',
                       val_start='2021-01-01', val_end='2023-01-01'):
    data = yf.download(ticker, start=train_start, end=val_end)

    training_data = data['Close'][train_start:train_end]
    validation_data = data['Close'][val_start:val_end]

    return training_data, validation_data, data


def build_and_train_model(training_data, validation_data):
    print("Finding best ARIMA parameters...")

    # TODO: use auto_arima from pmdarima
    order = (5, 1, 0)  # p, d, q parameters for ARIMA

    print(f"Fitting ARIMA{order} model...")
    model = ARIMA(training_data, order=order)
    model_fit = model.fit()

    print("Making predictions on validation set...")
    predictions = model_fit.forecast(steps=len(validation_data))

    mse = mean_squared_error(validation_data, predictions)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE: {rmse:.2f}")

    return model_fit, predictions


def print_monthly_predictions(predictions, future_dates, ticker):
    print(f"\n{ticker} Monthly Price Predictions:")
    print("-" * 40)
    print(f"{'Month':<15} {'Predicted Price':<15}")
    print("-" * 40)

    # convert predictions to a list if it's a Series or other iterable
    if hasattr(predictions, 'tolist'):
        pred_list = predictions.tolist()
    else:
        pred_list = list(predictions)

    for i in range(len(future_dates)):
        month_str = future_dates[i].strftime('%b %Y')
        if i < len(pred_list):
            price_str = f"${pred_list[i]:.2f}"
            print(f"{month_str:<15} {price_str:<15}")
        else:
            print(f"{month_str:<15} No prediction available")

def predict_next_12_months(model, historical_data):
    current_date = datetime.now()
    future_dates = []

    predictions = model.forecast(steps=12)

    for i in range(12):
        future_date = (current_date + timedelta(days=30 * (i + 1)))
        future_dates.append(future_date)

    return predictions, future_dates


def run_stock_prediction(ticker='AAPL'):
    print(f"Loading and preparing {ticker} data...")
    training_data, validation_data, historical_data = prepare_stock_data(ticker)

    print("Building and training ARIMA model...")
    model, val_predictions = build_and_train_model(training_data, validation_data)

    print("Refitting model on all available data for future predictions...")
    all_data = historical_data['Close']
    final_model = ARIMA(all_data, order=(5, 1, 0))
    final_model_fit = final_model.fit()

    print("Predicting prices for the next 12 months...")
    predictions, future_dates = predict_next_12_months(final_model_fit, historical_data)
    print_monthly_predictions(predictions, future_dates, ticker)

    return predictions, future_dates


print("Running prediction for SPY...")
spy_predictions, spy_dates = run_stock_prediction('SPY')

print("\nRunning prediction for AAPL...")
aapl_predictions, aapl_dates = run_stock_prediction('AAPL')

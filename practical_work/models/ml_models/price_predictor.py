
import numpy as np

print("My numpy version is: ", np.__version__)

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import yfinance as yf
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
from keras import models
from keras import layers
from datetime import datetime, timedelta


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df


def prepare_data(data, look_back=60):
    """
    Prepare data for LSTM model
    """
    prices = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(look_back, len(prices_scaled)):
        X.append(prices_scaled[i - look_back:i, 0])
        y.append(prices_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def create_lstm_model(look_back):
    """
    Create and compile LSTM model
    """
    model = models.Sequential([
        layers.LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        layers.Dropout(0.2),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def predict_future(model, last_sequence, scaler, num_days):
    """
    Predict future stock prices
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(num_days):
        current_sequence_reshaped = current_sequence.reshape(1, -1, 1)
        next_day_scaled = model.predict(current_sequence_reshaped)

        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_day_scaled

        future_predictions.append(next_day_scaled[0])

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions.flatten()


def main():
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    look_back = 60  # Number of days to look back

    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, start_date, end_date)

    X, y, scaler = prepare_data(df, look_back)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print("Training LSTM model...")
    model = create_lstm_model(look_back)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    days_to_predict = int(input("Enter number of days to predict: "))
    last_sequence = X[-1]
    future_predictions = predict_future(model, last_sequence, scaler, days_to_predict)

    last_date = datetime.strptime(end_date, '%Y-%m-%d')
    future_dates = [last_date + timedelta(days=x + 1) for x in range(days_to_predict)]

    print("\nPredicted prices for the next", days_to_predict, "days:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.date()}: ${price:.2f}")


if __name__ == "__main__":
    main()
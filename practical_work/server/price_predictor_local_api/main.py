from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

import os
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])
        return out

class LSTMWrapper:
    def __init__(self, lstm_model, x_scaler, y_scaler, device):
        self.model = lstm_model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device

    def predict(self, X):
        X_scaled = self.x_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(preds)

# def get_stock_data(ticker: str, period="2y"):
#     df = yf.download(ticker, period=period)
#     df['Return'] = df['Close'].pct_change()
#     df['Volatility'] = df['Return'].rolling(window=10).std()
#     df = df[['Close', 'Volume', 'Volatility']].dropna()
#     return df
import pandas as pd
import os


def get_stock_data(ticker: str, period="2y"):
    try:
        #raise ValueError("Simulated failure for fallback test")
        df = yf.download(ticker, period=period)
        if df.empty:
            raise ValueError("Empty data from Yahoo Finance.")
    except Exception as e:
        print(f"[WARNING] Failed to fetch data from yfinance for {ticker}: {e}")
        safe_ticker = ticker.replace("^", "")
        path = fr"E:\thesis_fallback_datasets\{safe_ticker}_processed.csv"
        if os.path.exists(path):
            print(f"Loading fallback data from {path}")

            df = pd.read_csv(
                path,
                skiprows=3,
                names=["Date", "Close", "Volume", "Volatility"],
                parse_dates=["Date"],
                index_col="Date"
            )

            end_date = df.index.max()
            if "y" in period:
                years = int(period.rstrip("y"))
                start_date = end_date - pd.DateOffset(years=years)
                df = df.loc[start_date:end_date]
            else:
                print("UNSUPPORTED PERIOD")
                raise ValueError("Unsupported period format for fallback.")
        else:
            print("NO FALLBACK DATA FOUND")
            raise FileNotFoundError(f"No fallback data found for {ticker}")

    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df = df[['Close', 'Volume', 'Volatility']].dropna()
    return df


def create_dataset(data, time_steps=20):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def train_lstm_model(X, y):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    model.train()
    for epoch in range(20):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return LSTMWrapper(model, X_scaler, y_scaler, device)

def predict_next_n_days(model_wrapper, last_sequence, days=10, time_steps=20):
    predictions = []
    for _ in range(days):
        input_seq = last_sequence.reshape(1, time_steps, -1)
        next_pred = model_wrapper.predict(input_seq)

        predicted_price = max(next_pred[0][0], 0.001)  # minimum threshold
        predictions.append(predicted_price)

        next_row = np.array([predicted_price, last_sequence[-1, 1], last_sequence[-1, 2]])
        last_sequence = np.vstack([last_sequence[1:], next_row])
    return predictions


def apply_daily_volatility_with_trend_conservation(predicted_prices, seed=None, pct_range=(0, 0.01)):
    if seed is not None:
        np.random.seed(seed)

    predicted_prices = np.array(predicted_prices)
    print(predicted_prices)
    adjusted_prices = [max(predicted_prices[0], np.random.uniform(0.0001, 0.001))]
    print(adjusted_prices)

    for i in range(1, len(predicted_prices)):
        prev = adjusted_prices[-1]
        predicted_diff = predicted_prices[i] - predicted_prices[i - 1]

        trend = np.sign(predicted_diff)

        if trend == 0:
            trend = np.random.choice([-1, 1])

        percentage_change = np.random.uniform(*pct_range)
        new_price = prev * (1 + trend * percentage_change)

        new_price = max(new_price, np.random.uniform(0.0001, 0.001))  # minimum threshold
        adjusted_prices.append(new_price)

    return adjusted_prices

def apply_daily_volatility_without_trend_conservation(predicted_prices, seed=None, pct_range=(0, 0.01)):
    if seed is not None:
        np.random.seed(seed)

    predicted_prices = np.array(predicted_prices)
    adjusted_prices = [predicted_prices[0]]

    for i in range(1, len(predicted_prices)):
        direction = np.random.choice([-1, 1])  # random up/down
        percentage_change = np.random.uniform(*pct_range)
        new_price = adjusted_prices[-1] * (1 + direction * percentage_change)
        new_price = max(new_price, 0.01)  # enforce minimum threshold
        adjusted_prices.append(new_price)

    return adjusted_prices


def apply_sentiment_adjustment(prices, sentiment_credibility_adjusted_score, decay_rate=0.2):
    sentiment_percentage = sentiment_credibility_adjusted_score / 10
    adjusted_prices = []

    for i in range(len(prices)):
        decay = decay_rate ** i

        adjustment_factor = 1 + sentiment_percentage * decay

        if i == 0:
            new_price = prices[0] * adjustment_factor
        else:
            new_price = adjusted_prices[-1] * adjustment_factor

            raw_trend = np.sign(prices[i] - prices[i - 1])
            new_trend = np.sign(new_price - adjusted_prices[-1])

            if raw_trend != new_trend and raw_trend != 0:
                adjustment_factor = 1 - sentiment_percentage * decay
                new_price = adjusted_prices[-1] * adjustment_factor

        new_price = max(new_price, np.random.uniform(0.0001, 0.001))
        adjusted_prices.append(new_price)

    return adjusted_prices


@app.get("/predict_stock/")
async def predict_stock(
        ticker: str = Query(..., description="Stock ticker symbol"),
        adjusted_sentiment_credibility_score: float = Query(..., description="Sentiment score in interval (-1, 1)")
        ):
    try:
        time_steps = 20
        df = get_stock_data(ticker)
        current_date = df.index[-1].date()
        #current_price = round(float(df['Close'].iloc[-1]), 2)
        current_price = round(df['Close'].iloc[-1].item(), 2)
        print(current_price)
        logging.info(f"Current price: {current_price}")

        values = df.values
        X, y = create_dataset(values, time_steps)
        model_wrapper = train_lstm_model(X, y)

        last_sequence = values[-time_steps:]
        predicted_prices = predict_next_n_days(model_wrapper, last_sequence, days=10, time_steps=time_steps)

        if adjusted_sentiment_credibility_score == 0.0:
            adjusted_volatility_prices = apply_daily_volatility_with_trend_conservation(predicted_prices)
            sentiment_adjusted_prices = predicted_prices
        else:
            sentiment_adjusted_prices = apply_sentiment_adjustment(predicted_prices,
                                                                   adjusted_sentiment_credibility_score)
            adjusted_volatility_prices = apply_daily_volatility_with_trend_conservation(sentiment_adjusted_prices)

        # sentiment_adjusted_prices = apply_sentiment_adjustment(predicted_prices, adjusted_sentiment_credibility_score)
        # adjusted_volatility_prices = apply_daily_volatility_with_trend_conservation(sentiment_adjusted_prices)


        last_date = df.index[-1]
        future_dates = [(last_date + timedelta(days=i + 1)) for i in range(10)]

        return {
            "ticker": ticker.upper(),
            "current_date": str(current_date),
            "current_price": current_price,
            "raw_predictions": [round(float(price), 2) for price in predicted_prices],
            "sentiment_predictions": [round(float(price), 2) for price in sentiment_adjusted_prices],
            "predictions": [
                {"date": str(date.date()), "predicted_close": round(float(price), 2)}
                for date, price in zip(future_dates, adjusted_volatility_prices)
            ]
        }
    except Exception as e:
        return {"error": str(e)}
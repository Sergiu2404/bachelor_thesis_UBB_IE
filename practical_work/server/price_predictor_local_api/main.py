# from fastapi import FastAPI, Query
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import MinMaxScaler
# from datetime import timedelta
#
#
# app = FastAPI()
#
# class LSTMPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size=128):
#         super().__init__()
#         self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.dropout1 = nn.Dropout(0.2)
#         self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
#         self.dropout2 = nn.Dropout(0.2)
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x = self.dropout1(x)
#         x, _ = self.lstm2(x)
#         x = self.dropout2(x)
#         return self.fc(x[:, -1, :])
#
#
# class LSTMWrapper:
#     def __init__(self, model, x_scaler, y_scaler):
#         self.model = model
#         self.x_scaler = x_scaler
#         self.y_scaler = y_scaler
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def predict(self, X):
#         X_scaled = self.x_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
#         X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
#         self.model.eval()
#         with torch.no_grad():
#             preds = self.model(X_tensor).cpu().numpy()
#         return self.y_scaler.inverse_transform(preds)
#
# def get_stock_data(ticker: str, period="2y"):
#     df = yf.download(ticker, period=period)
#     return df[['Close']].dropna()
#
#
# def create_dataset(data, time_steps=15):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)
#
# def train_model(X, y):
#     x_scaler = MinMaxScaler()
#     y_scaler = MinMaxScaler()
#
#     X_scaled = x_scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
#     y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
#
#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#     y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
#
#     dataset = TensorDataset(X_tensor, y_tensor)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LSTMPredictor(input_size=1).to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     model.train()
#     for epoch in range(20):
#         for X_batch, y_batch in dataloader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()
#
#     return LSTMWrapper(model, x_scaler, y_scaler)
#
# def predict_next_days(model_wrapper, last_data, days=10, time_steps=15):
#     predictions = []
#     current_sequence = last_data[-time_steps:]
#
#     for _ in range(days):
#         input_seq = current_sequence.reshape(1, time_steps, 1)
#         next_price = model_wrapper.predict(input_seq)[0][0]
#         predictions.append(next_price)
#         current_sequence = np.append(current_sequence[1:], [[next_price]], axis=0)
#
#     return predictions
#
# @app.get("/predict_stock/")
# async def predict_stock(ticker: str = Query(..., description="Stock ticker symbol")):
#     try:
#         df = get_stock_data(ticker)
#         data = df['Close'].values.reshape(-1, 1)
#         time_steps = 15
#
#         X, y = create_dataset(data, time_steps)
#         model_wrapper = train_model(X, y)
#
#         future_prices = predict_next_days(model_wrapper, data, days=10, time_steps=time_steps)
#         future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=10, freq='B')
#
#         return {
#             "ticker": ticker.upper(),
#             "predictions": [
#                 {"date": str(date.date()), "predicted_close": round(float(price), 2)}
#                 for date, price in zip(future_dates, future_prices)
#             ]
#         }
#     except Exception as e:
#         return {"error": str(e)}
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}


from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

app = FastAPI()


class FastLSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x


class LSTMWrapper:
    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, X):
        X_scaled = self.x_scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(preds)


def get_stock_data(ticker: str, period="2y"):
    df = yf.download(ticker, period=period)
    return df[['Close']].dropna()


def create_dataset(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


def add_noise_to_predictions(predictions, volatility_factor=0.02):
    noise = np.random.normal(0, predictions.mean() * volatility_factor, size=predictions.shape)
    return predictions + noise


def train_model(X, y):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Use 50% of data for faster training
    indices = np.random.choice(len(dataset), size=len(dataset) // 2, replace=False)
    dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastLSTM(hidden_size=64).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(15):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return LSTMWrapper(model, x_scaler, y_scaler)


def predict_next_days(model_wrapper, last_data, days=10, time_steps=30):
    predictions = []
    current_sequence = last_data[-time_steps:].copy()

    for _ in range(days):
        input_seq = current_sequence.reshape(1, time_steps, 1)
        next_price = model_wrapper.predict(input_seq)[0][0]
        predictions.append(next_price)
        current_sequence = np.append(current_sequence[1:], [[next_price]], axis=0)

    predictions = np.array(predictions).reshape(-1, 1)

    # Calculate momentum factor from recent prices
    price_diff = (current_sequence[-1, 0] - current_sequence[-7, 0]) / current_sequence[-7, 0]
    momentum = 1 + (price_diff * 0.5)

    # Apply momentum to predictions to maintain trends
    predictions = predictions * momentum

    # Add controlled noise to predictions for realism
    predictions = add_noise_to_predictions(predictions)

    return predictions.flatten()


@app.get("/predict_stock/")
async def predict_stock(ticker: str = Query(..., description="Stock ticker symbol")):
    try:
        df = get_stock_data(ticker)
        data = df['Close'].values.reshape(-1, 1)
        time_steps = 30

        X, y = create_dataset(data, time_steps)
        model_wrapper = train_model(X, y)

        future_prices = predict_next_days(
            model_wrapper,
            data,
            days=10,
            time_steps=time_steps
        )

        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=15, freq='B')[:10]

        historical_dates = [str(date.date()) for date in df.index[-20:]]

        return {
            "ticker": ticker.upper(),
            "predictions": [
                {"date": str(date.date()), "predicted_close": round(float(price), 2)}
                for date, price in zip(future_dates, future_prices)
            ],
            "historical": {
                "dates": historical_dates,
                "prices": [round(float(price), 2) for price in df['Close'].iloc[-20:].values]
            }
        }
    except Exception as e:
        return {"error": str(e)}
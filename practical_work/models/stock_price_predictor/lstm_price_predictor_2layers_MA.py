import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
from datetime import timedelta
import math


def get_stock_data(ticker, period="2y"):
    df = yf.download(ticker, period=period)

    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df[['Close', 'Volume', 'RSI']].dropna()


def create_dataset(data, time_steps=20):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)


class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BiLSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_model(X_train, y_train):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    input_shape = X_train.shape
    X_scaled = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    y_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        X_tensor, y_tensor, test_size=0.2, shuffle=False
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPredictor(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(20):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor.to(device))
            val_loss = criterion(val_output, y_val_tensor.to(device)).item()

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_model_state)

    class LSTMWrapper:
        def __init__(self, lstm_model, x_scaler, y_scaler, sequence_length):
            self.model = lstm_model
            self.x_scaler = x_scaler
            self.y_scaler = y_scaler
            self.device = device
            self.sequence_length = sequence_length

        def predict(self, X):
            X_scaled = self.x_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(X_tensor).cpu().numpy()
            return self.y_scaler.inverse_transform(preds)

    return LSTMWrapper(model, X_scaler, y_scaler, input_shape[1])


def evaluate_model(y_true, y_pred):
    metrics = {}
    mse = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = math.sqrt(mse)
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    metrics['MAPE'] = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    metrics['R2'] = r2_score(y_true, y_pred)

    actual_directions = np.sign(np.diff(y_true.reshape(-1)))
    predicted_directions = np.sign(np.diff(y_pred.reshape(-1)))
    valid_indices = np.where(actual_directions != 0)[0]
    if len(valid_indices) > 0:
        correct_directions = np.sum(actual_directions[valid_indices] == predicted_directions[valid_indices])
        metrics['Directional Accuracy'] = correct_directions / len(valid_indices) * 100
    else:
        metrics['Directional Accuracy'] = np.nan

    return metrics

if __name__ == "__main__":
    ticker = "AAPL"
    time_steps = 20
    prediction_days = 10

    # Step 1: Load and preprocess data
    df = get_stock_data(ticker)
    values = df.values
    X, y = create_dataset(values, time_steps)

    # Step 2: Train/Test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Step 3: Train model
    start_time = time.time()
    model = train_lstm_model(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.4f} seconds.")

    # Step 4: Predict on test set
    y_pred = model.predict(X_test)

    # Step 5: Evaluate model
    metrics = evaluate_model(y_test, y_pred)
    print("\nEvaluation Metrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Step 6: Predict future 10 days
    last_sequence = values[-time_steps:]
    future_predictions = []

    for _ in range(prediction_days):
        input_seq = last_sequence.reshape(1, time_steps, values.shape[1])
        next_pred = model.predict(input_seq)
        next_close = next_pred[0][0]
        future_predictions.append(next_close)

        # Use the last known RSI and Volume for next input
        next_step = np.array([next_close, last_sequence[-1][1], last_sequence[-1][2]])
        last_sequence = np.vstack([last_sequence[1:], next_step])

    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=prediction_days)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
    future_df.set_index('Date', inplace=True)

    print(f"\nPredicted prices for {ticker} for the next {prediction_days} days:")
    print(future_df)
# Evaluation Metrics on Test Set:
# RMSE: 8.0657
# MAE: 6.1242
# MAPE: 2.8843
# R2: 0.7740, Training completed in 2.3218 seconds.

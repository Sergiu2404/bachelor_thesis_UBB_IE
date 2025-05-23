# import time
#
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from datetime import timedelta
# import math
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
#
# def get_stock_data(ticker, period="1y"):
#     df = yf.download(ticker, period=period)
#     return df[['Close', 'Volume']].dropna()
#
#
# def create_dataset(data, time_steps=10):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps, 0])
#     return np.array(X), np.array(y)
#
#
# class LSTMPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size=32):
#         super(LSTMPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.dropout(out)
#         out = self.fc(out[:, -1, :])
#         return out
#
#
#
# def train_lstm_model(X_train, y_train):
#     X_scaler = MinMaxScaler()
#     y_scaler = MinMaxScaler()
#
#     X_scaled = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
#     y_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
#
#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#     y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
#
#     X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
#         X_tensor, y_tensor, test_size=0.2, shuffle=False
#     )
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LSTMPredictor(input_size=X_train.shape[2], hidden_size=32).to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
#
#     best_loss = float('inf')
#     patience = 8
#     counter = 0
#
#     for epoch in range(30):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()
#
#         model.eval()
#         with torch.no_grad():
#             val_output = model(X_val_tensor.to(device))
#             val_loss = criterion(val_output, y_val_tensor.to(device)).item()
#         scheduler.step(val_loss)
#
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_model_state = model.state_dict()
#             counter = 0
#         else:
#             counter += 1
#             if counter >= patience:
#                 break
#
#     model.load_state_dict(best_model_state)
#
#     class LSTMWrapper:
#         def __init__(self, lstm_model, x_scaler, y_scaler):
#             self.model = lstm_model
#             self.x_scaler = x_scaler
#             self.y_scaler = y_scaler
#             self.device = device
#
#         def predict(self, X):
#             X_scaled = self.x_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
#             X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
#             self.model.eval()
#             with torch.no_grad():
#                 preds = self.model(X_tensor).cpu().numpy()
#             return self.y_scaler.inverse_transform(preds)
#
#     return LSTMWrapper(model, X_scaler, y_scaler)
#
#
#
# def predict_next_n_days(ticker, days=10, time_steps=15):
#     df = get_stock_data(ticker)
#     values = df.values
#     X, y = create_dataset(values, time_steps)
#     model = train_lstm_model(X, y)
#
#     last_sequence = values[-time_steps:]
#     predictions = []
#
#     for _ in range(days):
#         input_seq = last_sequence.reshape(1, time_steps, values.shape[1])
#         next_pred = model.predict(input_seq)
#         predictions.append(next_pred[0][0])
#
#         next_step = np.append(next_pred[0][0], last_sequence[-1, 1])  # use last known Volume
#         last_sequence = np.vstack([last_sequence[1:], next_step])
#
#     last_date = df.index[-1]
#     future_dates = [(last_date + timedelta(days=i + 1)) for i in range(days)]
#
#     future_df = pd.DataFrame({
#         'Date': future_dates,
#         'Predicted Close': predictions
#     })
#     future_df.set_index('Date', inplace=True)
#     return future_df
#
#
# def evaluate_model(y_true, y_pred):
#     metrics = {}
#     mse = mean_squared_error(y_true, y_pred)
#     metrics['RMSE'] = math.sqrt(mse)
#     metrics['MAE'] = mean_absolute_error(y_true, y_pred)
#     y_true_flat = y_true.flatten()
#     y_pred_flat = y_pred.flatten()
#     metrics['MAPE'] = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
#     metrics['R2'] = r2_score(y_true, y_pred)
#
#     actual_directions = np.sign(np.diff(y_true.reshape(-1)))
#     predicted_directions = np.sign(np.diff(y_pred.reshape(-1)))
#
#     valid_indices = np.where(actual_directions != 0)[0]
#     if len(valid_indices) > 0:
#         correct_directions = np.sum(actual_directions[valid_indices] == predicted_directions[valid_indices])
#         metrics['Directional Accuracy'] = correct_directions / len(valid_indices) * 100
#     else:
#         metrics['Directional Accuracy'] = np.nan
#
#     return metrics
#
#
# if __name__ == "__main__":
#     ticker = "AAPL"
#     predictions = predict_next_n_days(ticker, days=10)
#     print(f"\nPredicted prices for {ticker} for the next 10 days:")
#     print(predictions)
#
#     df = get_stock_data(ticker)
#     values = df.values
#     time_steps = 15
#
#     start_train = time.time()
#     X, y = create_dataset(values, time_steps)
#     split_index = int(len(X) * 0.8)
#     X_train, X_test = X[:split_index], X[split_index:]
#     y_train, y_test = y[:split_index], y[split_index:]
#
#     model = train_lstm_model(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     print(f"{time.time() - start_train}s took to train and predict")
#
#     metrics = evaluate_model(y_test, y_pred)
#     print("\nEvaluation Metrics on Historical Test Set:")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")


# RMSE: 11.1416
# MAE: 9.4016
# MAPE: 4.1635
# R2: 0.5641
# # Directional Accuracy: 57.5342






import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class StockDataset(Dataset):
    def __init__(self, features, seq_len=30):
        self.X = []
        self.y = []
        for i in range(seq_len, len(features)):
            self.X.append(features[i-seq_len:i])
            self.y.append(features[i][0])  # Predicting the next "Close" price (index 0)
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y)).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def prepare_data(df, seq_len=30):
    df['MA'] = df['Close'].rolling(window=5).mean()
    features = df[['Close', 'MA']].copy().dropna()
    # features.dropna(inplace=True)
    features = features.values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    if len(scaled) <= seq_len:
        raise ValueError("Not enough data to create any sequences. Try reducing sequence length.")

    dataset = StockDataset(scaled, seq_len)
    return dataset, scaler

def train_model(df, epochs=50, seq_len=30, batch_size=32):
    dataset, scaler = prepare_data(df, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Total samples: {len(dataset)}")

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.6f}")
    return model, scaler

def predict_next(df, model, scaler, seq_len=30):
    df['MA'] = df['Close'].rolling(window=5).mean()
    df = df[['Close', 'MA']].dropna()

    if len(df) < seq_len:
        raise ValueError("Not enough data to make prediction. Need at least `seq_len` rows after MA calculation.")

    features = df.values
    scaled = scaler.transform(features)
    seq = torch.FloatTensor(scaled[-seq_len:].reshape(1, seq_len, 2))

    model.eval()
    with torch.no_grad():
        pred = model(seq).item()

    inverse = scaler.inverse_transform([[0, pred]])[0][1]
    return inverse


df = yf.download("AAPL", period="3y", interval="1d")
model, scaler = train_model(df, seq_len=20)
next_price = predict_next(df, model, scaler)
print(f"Next predicted price: ${next_price:.2f}")

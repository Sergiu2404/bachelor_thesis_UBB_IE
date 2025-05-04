# # Full PyTorch-based rewrite of your LSTM stock prediction script
#
# import numpy as np
# import pandas as pd
# import yfinance as yf
# import datetime
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import os
# import warnings
# import requests
#
# warnings.filterwarnings('ignore')
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")
#
# model_path_dir = r"E:\\saved_models\\LSTM_price_prediction_model"
# os.makedirs(model_path_dir, exist_ok=True)
# model_path_file = os.path.join(model_path_dir, "LSTM_price_prediction_model.pt")
#
# session = requests.Session()
# ticker = yf.Ticker("^GSPC", session=session)
# data = ticker.history(period="5y").reset_index()
#
# data['Date'] = pd.to_datetime(data['Date'])
# stock_close = data.filter(['Close'])
# dataset = stock_close.values
# train_data_len = int(np.ceil(len(dataset) * 0.95))
#
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(dataset)
# training_data = scaled_data[:train_data_len]
#
# sequence_length = 60
# X_train, y_train = [], []
# for i in range(sequence_length, len(training_data)):
#     X_train.append(training_data[i - sequence_length:i, 0])
#     y_train.append(training_data[i, 0])
#
# X_train, y_train = np.array(X_train), np.array(y_train)
# X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
# y_train = torch.tensor(y_train, dtype=torch.float32)
#
# class LSTMModel(nn.Module):
#     def __init__(self):
#         super(LSTMModel, self).__init__()
#         self.lstm1 = nn.LSTM(input_size=1, hidden_size=100, batch_first=True, bidirectional=True)
#         self.dropout1 = nn.Dropout(0.2)
#         self.lstm2 = nn.LSTM(input_size=200, hidden_size=100, batch_first=True, bidirectional=True)
#         self.dropout2 = nn.Dropout(0.2)
#         self.lstm3 = nn.LSTM(input_size=200, hidden_size=100, batch_first=True, bidirectional=True)
#         self.dropout3 = nn.Dropout(0.2)
#         self.fc1 = nn.Linear(200, 50)
#         self.relu1 = nn.ReLU()
#         self.dropout4 = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(50, 25)
#         self.fc3 = nn.Linear(25, 1)
#
#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x = self.dropout1(x)
#         x, _ = self.lstm2(x)
#         x = self.dropout2(x)
#         x, _ = self.lstm3(x)
#         x = self.dropout3(x)
#         x = x[:, -1, :]
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout4(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x.squeeze()
#
# model = LSTMModel().to(DEVICE)
#
# if os.path.exists(model_path_file):
#     print("Model found. Loading...")
#     model.load_state_dict(torch.load(model_path_file))
# else:
#     print("No model found. Training...")
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#     best_loss = float('inf')
#     patience = 10
#     wait = 0
#
#     for epoch in range(20):
#         model.train()
#         optimizer.zero_grad()
#         output = model(X_train.to(DEVICE))
#         loss = criterion(output, y_train.to(DEVICE))
#         loss.backward()
#         optimizer.step()
#
#         print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.6f}")
#
#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             wait = 0
#             torch.save(model.state_dict(), model_path_file)
#         else:
#             wait += 1
#             if wait >= patience:
#                 print("Early stopping.")
#                 break
#
# model.eval()
# test_data = scaled_data[train_data_len - sequence_length:]
# X_test, y_test = [], dataset[train_data_len:]
# for i in range(sequence_length, len(test_data)):
#     X_test.append(test_data[i - sequence_length:i, 0])
#
# X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
# with torch.no_grad():
#     predictions = model(X_test).cpu().numpy()
#
# predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
# rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#
# # Future predictions
# last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
# current_batch = torch.tensor(last_sequence, dtype=torch.float32).to(DEVICE)
#
# future_predictions = []
# model.eval()
# for _ in range(10):
#     current_pred = model(current_batch)
#     pred_value = current_pred.squeeze().item()
#     future_predictions.append(pred_value)
#     next_input = torch.tensor([[[pred_value]]], dtype=torch.float32, device=current_batch.device)
#     current_batch = torch.cat((current_batch[:, 1:, :], next_input), dim=1)
#
#
# future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
#
# last_date = data['Date'].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')
#
# future_df = pd.DataFrame({
#     'Date': future_dates,
#     'Predicted_Close': future_prices.flatten()
# })
#
# print("\nPredictions for the next 10 business days:")
# for i, (date, price) in enumerate(zip(future_dates, future_prices.flatten())):
#     print(f"Day {i + 1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
#
# test = data[train_data_len:].copy()
# test['Predictions'] = np.nan
# min_len = min(len(test), len(predictions))
# test.iloc[:min_len, test.columns.get_loc('Predictions')] = predictions.flatten()[:min_len]
#
# mae = mean_absolute_error(test['Close'][:min_len], test['Predictions'][:min_len])
# mse = mean_squared_error(test['Close'][:min_len], test['Predictions'][:min_len])
# rmse = np.sqrt(mse)
# r2 = r2_score(test['Close'][:min_len], test['Predictions'][:min_len])
#
# print("\nModel Performance Metrics:")
# print(f"Mean Absolute Error (MAE): ${mae:.2f}")
# print(f"Mean Squared Error (MSE): ${mse:.2f}")
# print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
# print(f"R-squared (R²): {r2:.4f}")
# accuracy = 100 - (mae / test['Close'][:min_len].mean() * 100)
# print(f"Approximate Prediction Accuracy: {accuracy:.2f}%")
#
# future_df.to_csv('future_stock_predictions.csv', index=False)
# test[['Date', 'Close', 'Predictions']].to_csv('test_predictions.csv', index=False)







# import numpy as np
# import pandas as pd
# import yfinance as yf
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import time
# import os
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model_path_file = "E:\saved_models\LSTM_price_prediction_model\LSTM_price_prediction_model.pt"
#
# ticker = yf.Ticker("^GSPC")
# data = ticker.history(start="2021-07-01", end="2024-02-01").reset_index()
#
# data['Date'] = pd.to_datetime(data['Date'])
# dataset = data[['Close']].values
#
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(dataset)
#
# sequence_length = 60
# train_size = int(len(scaled_data) * 0.85)
#
# train_data = scaled_data[:train_size]
# test_data = scaled_data[train_size - sequence_length:]
#
# def create_sequences(data, seq_len):
#     X, y = [], []
#     for i in range(seq_len, len(data)):
#         X.append(data[i - seq_len:i, 0])
#         y.append(data[i, 0])
#     return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)
#
#
# X_train, y_train = create_sequences(train_data, sequence_length)
# X_test, y_test = create_sequences(test_data, sequence_length)
#
# X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
# X_test = X_test.to(DEVICE)
#
# class SimpleLSTM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
#         self.fc1 = nn.Linear(64, 128)
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(128, 1)
#
#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x = x[:, -1, :]
#         x = self.fc1(x)
#         x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x.squeeze()
#
# model = SimpleLSTM().to(DEVICE)
#
# if os.path.exists(model_path_file):
#     model.load_state_dict(torch.load(model_path_file))
# else:
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()
#     best_loss = float('inf')
#     patience = 10
#     wait = 0
#     for epoch in range(50):
#         model.train()
#         optimizer.zero_grad()
#         output = model(X_train)
#         loss = loss_fn(output, y_train)
#         loss.backward()
#         optimizer.step()
#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             wait = 0
#             #torch.save(model.state_dict(), model_path_file)
#         else:
#             wait += 1
#             if wait >= patience:
#                 break
#
# model.eval()
# with torch.no_grad():
#     predictions = model(X_test).cpu().numpy()
#
# predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
# true = dataset[train_size:]
# rmse = np.sqrt(np.mean((predictions - true[:len(predictions)]) ** 2))
# print(f"RMSE: {rmse:.2f}")
#
# last_sequence = torch.tensor(scaled_data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
# future_predictions = []
#
# model.eval()
# for _ in range(10):
#     with torch.no_grad():
#         pred = model(last_sequence).item()
#     future_predictions.append(pred)
#     next_input = torch.tensor([[pred]], dtype=torch.float32).to(DEVICE)
#     last_sequence = torch.cat((last_sequence[:, 1:, :], next_input.unsqueeze(0)), dim=1)
#
# future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
# last_date = data['Date'].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')
#
# print("\nNext 10 business days predictions:")
# for date, price in zip(future_dates, future_prices.flatten()):
#     print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")











import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path_file = "E:\saved_models\LSTM_price_prediction_model\LSTM_price_prediction_model.pt"

ticker = yf.Ticker("^GSPC")
data = ticker.history(start="2025-01-01", end="2025-05-01").reset_index()

data['Date'] = pd.to_datetime(data['Date'])
dataset = data[['Close']].values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

sequence_length = 30
train_size = int(len(scaled_data) * 0.85)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - sequence_length:]

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
X_test = X_test.to(DEVICE)

class ImprovedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.norm = nn.LayerNorm(256)

        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x.squeeze()

start_time = time.time()

model = ImprovedLSTM().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")


if os.path.exists(model_path_file):
    model.load_state_dict(torch.load(model_path_file))
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    patience = 10
    wait = 0
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
            #torch.save(model.state_dict(), model_path_file)
        else:
            wait += 1
            if wait >= patience:
                break

model.eval()
with torch.no_grad():
    predictions = model(X_test).cpu().numpy()

predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
true = dataset[train_size:]
rmse = np.sqrt(np.mean((predictions - true[:len(predictions)]) ** 2))
print(f"RMSE: {rmse:.2f}")

last_sequence = torch.tensor(scaled_data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
future_predictions = []

model.eval()
for _ in range(10):
    with torch.no_grad():
        pred = model(last_sequence).item()
    future_predictions.append(pred)
    next_input = torch.tensor([[pred]], dtype=torch.float32).to(DEVICE)
    last_sequence = torch.cat((last_sequence[:, 1:, :], next_input.unsqueeze(0)), dim=1)

future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = data['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')

print("\nNext 10 business days predictions:")
for date, price in zip(future_dates, future_prices.flatten()):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

print(f"took {time.time() - start_time}")

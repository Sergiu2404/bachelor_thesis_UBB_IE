import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path_file = "E:\saved_models\LSTM_price_prediction_model\LSTM_price_prediction_model.pt"

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365.25 * 30)

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

data = yf.download("^GSPC", start=start_date_str, end=end_date_str, interval='1mo', auto_adjust=True).reset_index()

data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

print(data.columns)

data['Date_'] = pd.to_datetime(data['Date_'])
data.set_index('Date_', inplace=True)

monthly_data = data['Close_^GSPC'].resample('M').last().reset_index()

print(monthly_data.columns)

dataset = monthly_data[['Close_^GSPC']].values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

sequence_length = 12
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

# predict next 10 years (one rec / year)
last_sequence = torch.tensor(scaled_data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
future_predictions = []

model.eval()
for _ in range(10):  # predict the next 10 years
    with torch.no_grad():
        pred = model(last_sequence).item()
    future_predictions.append(pred)
    next_input = torch.tensor([[pred]], dtype=torch.float32).to(DEVICE)
    last_sequence = torch.cat((last_sequence[:, 1:, :], next_input.unsqueeze(0)), dim=1)

future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = monthly_data['Date_'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=365), periods=10, freq='A')  # frequency A for year-end

print("\nNext 10 years predictions:")
for date, price in zip(future_dates, future_prices.flatten()):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

print(f"took {time.time() - start_time}")

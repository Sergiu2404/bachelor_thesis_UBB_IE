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

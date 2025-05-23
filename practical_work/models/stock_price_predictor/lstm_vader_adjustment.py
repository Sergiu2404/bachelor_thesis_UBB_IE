import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class PriceSentimentDataset(Dataset):
    def __init__(self, prices, sentiments, target, seq_len=30):
        self.X = []
        self.S = []
        self.y = []
        for i in range(seq_len, len(prices)):
            self.X.append(prices[i-seq_len:i])
            self.S.append(sentiments[i-seq_len:i])
            self.y.append(target[i])
        self.X = torch.FloatTensor(self.X)
        self.S = torch.FloatTensor(self.S)
        self.y = torch.FloatTensor(self.y).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx], self.y[idx]


class LSTMSentimentModel(nn.Module):
    def __init__(self, input_dim=1, sentiment_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.price_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.sent_lstm = nn.LSTM(sentiment_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_price, x_sent):
        _, (h_price, _) = self.price_lstm(x_price)
        _, (h_sent, _) = self.sent_lstm(x_sent)
        h_combined = torch.cat((h_price[-1], h_sent[-1]), dim=1)
        return self.fc(h_combined)


def train_model(df, epochs=50, seq_len=30, batch_size=32):
    prices = df['Close'].values.reshape(-1, 1)
    sentiments = df['Sentiment'].values.reshape(-1, 1)

    # Normalize
    scaler_price = MinMaxScaler()
    scaler_sent = MinMaxScaler()
    prices_scaled = scaler_price.fit_transform(prices)
    sentiments_scaled = scaler_sent.fit_transform(sentiments)

    # Target is next day price
    target = prices_scaled[1:]
    prices_scaled = prices_scaled[:-1]
    sentiments_scaled = sentiments_scaled[:-1]

    dataset = PriceSentimentDataset(prices_scaled, sentiments_scaled, target, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMSentimentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, s, y in dataloader:
            optimizer.zero_grad()
            output = model(x, s)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.6f}")

    return model, scaler_price

def predict_with_news(df, model, scaler_price, seq_len=30, days=10, sentiment_score=0.0, decay=0.9):
    """
    Predict next `days` prices adjusted by news sentiment score using exponential decay.
    """
    model.eval()
    prices = df['Close'].values.reshape(-1, 1)
    scaled_prices = scaler_price.transform(prices)
    last_seq = scaled_prices[-seq_len:].reshape(1, seq_len, 1)
    last_seq = torch.FloatTensor(last_seq)

    # dummy sentiment sequence
    sentiment_seq = torch.zeros_like(last_seq)

    predicted_scaled_prices = []

    with torch.no_grad():
        current_price = prices[-1][0]
        for i in range(days):
            output = model(last_seq, sentiment_seq).item()
            # reverse scale
            predicted_price = scaler_price.inverse_transform([[output]])[0][0]

            # Sentiment adjustment
            weight = decay ** i  # decay factor
            sentiment_adjustment = current_price * (sentiment_score * weight * 0.1)  # 1% of sentiment * decay
            predicted_price += sentiment_adjustment

            predicted_scaled_prices.append(predicted_price)

            # update sequence with new prediction
            new_scaled = torch.FloatTensor(scaler_price.transform([[predicted_price]]))
            last_seq = torch.cat([last_seq[:, 1:], new_scaled.view(1, 1, 1)], dim=1)

    return predicted_scaled_prices



import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_sentiment(df):

    df['Sentiment'] = 0.0
    return df

df = yf.download("AAPL", period="1y", interval="1d")
df = fetch_sentiment(df)
df.dropna(inplace=True)

model, price_scaler = train_model(df)
# Assume sentiment score from a news headline: +0.8 (positive), -0.6 (negative)
news_sentiment_score = -0.7

predicted_prices = predict_with_news(df, model, price_scaler, sentiment_score=news_sentiment_score)

print("\nAdjusted predictions with sentiment:")
for i, p in enumerate(predicted_prices):
    print(f"Day {i+1}: ${p:.2f}")

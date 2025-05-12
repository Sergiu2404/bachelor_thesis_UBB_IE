import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class TemporalFusionTransformerModel(nn.Module):
    """
    Temporal Fusion Transformer for time series forecasting
    Based on the paper: https://arxiv.org/abs/1912.09363
    """

    def __init__(self, input_size, hidden_size=128, num_attention_heads=4, dropout=0.1, num_layers=2):
        super(TemporalFusionTransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # feature processing layers
        self.input_projection = nn.Linear(input_size, hidden_size)

        # LSTM layers for temporal processing
        self.encoder_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )

        # self-attention layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        # feed-forward network
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_size]

        # LSTM encoder
        x, _ = self.encoder_lstm(x)  # [batch_size, seq_len, hidden_size]

        # Self-attention with residual connection and layer norm
        residual = x
        x = self.norm1(x)

        # Transpose for attention: [seq_len, batch_size, hidden_size]
        x = x.transpose(0, 1)
        x, _ = self.self_attention(x, x, x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_size]

        x = residual + self.dropout1(x)

        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))

        # Output projection
        outputs = self.output_layer(x)  # [batch_size, seq_len, 1]

        return outputs


class StockDataset(Dataset):
    """PyTorch Dataset for stock data"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockPricePredictor:
    def __init__(self, input_window=60, output_window=10, batch_size=32, learning_rate=0.001):
        """
        Stock price prediction using Temporal Fusion Transformer

        Args:
            input_window: Size of lookback window (days)
            output_window: Size of prediction window (days)
            batch_size: Training batch size
            learning_rate: Model learning rate
        """
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _create_sequences(self, data):
        """Create input/output sequences for training and evaluation"""
        X, y = [], []

        # The data should have features like: Open, High, Low, Close, Volume
        for i in range(len(data) - self.input_window - self.output_window + 1):
            # Input sequence
            X.append(data[i:(i + self.input_window)].values)

            # Target sequence (next output_window closing prices)
            target_idx = i + self.input_window
            y.append(data[target_idx:target_idx + self.output_window]['Close'].values.reshape(-1, 1))

        return np.array(X), np.array(y)

    def get_data(self, ticker, start_date, end_date):
        """Download stock data"""
        data = yf.download(ticker, start=start_date, end=end_date)

        # Check if data was downloaded successfully
        if len(data) == 0:
            raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}")

        # Feature engineering
        # Add technical indicators that might help with prediction
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = self._compute_rsi(data['Close'], window=14)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()

        # Drop NaN values that appear due to rolling windows
        data = data.dropna()

        return data

    def _compute_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        delta = delta[1:]

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        down = down.abs()

        avg_gain = up.rolling(window=window).mean()
        avg_loss = down.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def preprocess_data(self, data):
        """Preprocess data for model input"""
        features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'Daily_Return', 'Volatility']]

        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_features_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

        # Scale target (Close prices) separately to easily inverse transform later
        target = data[['Close']]
        self.target_scaler.fit(target)

        return scaled_features_df

    def prepare_datasets(self, data, train_ratio=0.8):
        """Prepare training and testing datasets"""
        # Create sequences
        X, y = self._create_sequences(data)

        # Split into training and testing sets
        train_size = int(len(X) * train_ratio)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create PyTorch datasets
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, X_test, y_test

    def build_model(self):
        """Initialize the TFT model"""
        input_size = self.feature_scaler.n_features_in_  # Number of features
        self.model = TemporalFusionTransformerModel(input_size=input_size)
        self.model.to(self.device)
        return self.model

    def train(self, train_loader, test_loader, epochs=50, patience=10):
        """Train the model"""
        if self.model is None:
            self.build_model()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        counter = 0
        best_model = None

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)

                # We only want to predict the 'output_window' time steps
                outputs = outputs[:, -self.output_window:, :]

                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    outputs = outputs[:, -self.output_window:, :]
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model = self.model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)

        return train_losses, val_losses

    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test set"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        self.model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = self.model(X_test_tensor)
            y_pred = y_pred[:, -self.output_window:, 0].cpu().numpy()

        # Reshape predictions and actual values for inverse scaling
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_test_reshaped = y_test.reshape(-1, 1)

        # Inverse transform to get actual prices
        y_pred_orig = self.target_scaler.inverse_transform(y_pred_reshaped)
        y_test_orig = self.target_scaler.inverse_transform(y_test_reshaped)

        # Calculate metrics
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        r2 = r2_score(y_test_orig, y_pred_orig)
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

        print(f'Mean Absolute Error: {mae:.4f}')
        print(f'Root Mean Squared Error: {rmse:.4f}')
        print(f'R² Score: {r2:.4f}')
        print(f'Mean Absolute Percentage Error: {mape:.4f}%')

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

        return metrics, y_pred_orig, y_test_orig

    def predict_future(self, ticker, days=10):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Get today's date and format it
        today = datetime.now()
        end_date = today.strftime('%Y-%m-%d')

        # Calculate start date (input_window days before today)
        start_date = (today - timedelta(days=self.input_window * 2)).strftime('%Y-%m-%d')

        # Download the latest data
        latest_data = yf.download(ticker, start=start_date, end=end_date)

        # Apply the same feature engineering as in training
        latest_data['MA5'] = latest_data['Close'].rolling(window=5).mean()
        latest_data['MA20'] = latest_data['Close'].rolling(window=20).mean()
        latest_data['RSI'] = self._compute_rsi(latest_data['Close'], window=14)
        latest_data['Daily_Return'] = latest_data['Close'].pct_change()
        latest_data['Volatility'] = latest_data['Daily_Return'].rolling(window=20).std()

        # Drop NaN values
        latest_data = latest_data.dropna()

        if len(latest_data) < self.input_window:
            raise ValueError(f"Not enough data points. Need at least {self.input_window} days of data.")

        # Get the most recent input_window days of data
        recent_data = latest_data.tail(self.input_window)

        # Extract features and scale
        features = recent_data[
            ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'Daily_Return', 'Volatility']]
        scaled_features = self.feature_scaler.transform(features)

        # Make prediction
        X_pred = torch.tensor(scaled_features.reshape(1, self.input_window, -1), dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_pred)
            predictions = predictions[0, -self.output_window:, 0].cpu().numpy().reshape(-1, 1)

        # Inverse transform to get actual prices
        predicted_prices = self.target_scaler.inverse_transform(predictions).flatten()

        # Create dates for the prediction
        last_date = latest_data.index[-1]
        prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=self.output_window)

        # Create a DataFrame with the predictions
        forecast_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Close': predicted_prices
        })
        forecast_df.set_index('Date', inplace=True)

        return forecast_df, latest_data


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = StockPricePredictor(input_window=60, output_window=10)

    # Set ticker and date range
    ticker = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Get and preprocess data
    data = predictor.get_data(ticker, start_date, end_date)
    processed_data = predictor.preprocess_data(data)

    train_loader, test_loader, X_test, y_test = predictor.prepare_datasets(processed_data)

    predictor.build_model()
    train_losses, val_losses = predictor.train(train_loader, test_loader, epochs=50)

    metrics, y_pred, y_test = predictor.evaluate(X_test, y_test)

    forecast_df, latest_data = predictor.predict_future(ticker, days=10)
    print("\nPredicted prices for the next 10 days:")
    print(forecast_df)
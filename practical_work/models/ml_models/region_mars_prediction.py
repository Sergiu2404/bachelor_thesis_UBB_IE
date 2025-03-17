import os
import numpy as np
import pandas as pd
import yfinance as yf

#from pyearth import
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class MarsStockPredictor:
    def __init__(self, region, model_path):
        self.region = region
        self.model_path = model_path
        self.model_file = os.path.join(model_path, f"{region}_mars_model.joblib")
        self.scaler_file = os.path.join(model_path, f"{region}_scaler.joblib")
        self.model = None
        self.scaler = None
        self.training_index = None

        self.region_config = {
            'us': {'index': '^GSPC', 'name': 'S&P 500'},  # S&P 500
            'japan': {'index': '^N225', 'name': 'Nikkei 225'},  # Nikkei 225
            'german': {'index': '^GDAXI', 'name': 'DAX'}  # DAX
        }

        os.makedirs(model_path, exist_ok=True)
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                print(f"Loading existing model for {self.region} region...")
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                print("Model loaded successfully!")
                return True
            else:
                print(f"No existing model found for {self.region} region.")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _create_features(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['SMA_5_20_Ratio'] = df['SMA_5'] / df['SMA_20']
        df['SMA_20_50_Ratio'] = df['SMA_20'] / df['SMA_50']
        df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']

        df['ROC_5'] = df['Close'].pct_change(periods=5)
        df['ROC_10'] = df['Close'].pct_change(periods=10)
        df['ROC_20'] = df['Close'].pct_change(periods=20)

        df['Target_12m_Return'] = df['Close'].pct_change(periods=252).shift(-252)  # 252 trading days ≈ 1 year
        df.dropna(inplace=True)

        features = ['Returns', 'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
                    'Volatility_10', 'Volatility_30', 'RSI',
                    'SMA_5_20_Ratio', 'SMA_20_50_Ratio', 'SMA_50_200_Ratio',
                    'ROC_5', 'ROC_10', 'ROC_20']
        X = df[features]
        y = df['Target_12m_Return']

        return X, y

    def train(self, years=10, max_terms=30):
        if self.region not in self.region_config:
            raise ValueError(f"Invalid region: {self.region}. Must be one of {list(self.region_config.keys())}")

        index_symbol = self.region_config[self.region]['index']
        index_name = self.region_config[self.region]['name']

        print(f"Training model for {self.region} region using {index_name} ({index_symbol})...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        df = yf.download(index_symbol, start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"Could not download data for {index_symbol}")

        print(f"Downloaded {len(df)} days of historical data for {index_name}")

        X, y = self._create_features(df)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = Earth(max_terms=max_terms, max_degree=2, verbose=1)
        self.model.fit(X_scaled, y)

        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)

        print(f"\nModel training complete for {self.region} region")
        print(f"Selected {len(self.model.basis_)} basis functions")

        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(f"Training R² Score: {r2:.4f}")
        print(f"Training MSE: {mse:.6f}")

        self.training_index = index_symbol
        return self.model

    def predict(self, ticker, plot=True):
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded. Call train() first.")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)

        print(f"Downloading data for {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            raise ValueError(f"Could not download data for {ticker}")

        X, actual_returns = self._create_features(stock_data)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        current_price = stock_data['Close'].iloc[-1]

        latest_X = X.iloc[-1:].values
        latest_X_scaled = self.scaler.transform(latest_X)
        predicted_return = self.model.predict(latest_X_scaled)[0]
        predicted_price = current_price * (1 + predicted_return)

        print(f"\nPrediction for {ticker} using {self.region} model:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted 12-Month Return: {predicted_return * 100:.2f}%")
        print(f"Predicted Price in 12 Months: ${predicted_price:.2f}")

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(stock_data.index[len(stock_data) - len(predictions):],
                     actual_returns * 100,
                     label='Actual 12-Month Returns (%)',
                     color='blue')
            plt.plot(stock_data.index[len(stock_data) - len(predictions):],
                     predictions * 100,
                     label='Predicted 12-Month Returns (%)',
                     color='red')
            plt.title(f'MARS Model: Actual vs Predicted 12-Month Returns for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.legend()
            plt.grid(True)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.show()

        return predicted_return, current_price, predicted_price

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() first.")

        importance = self.model.summary()
        return importance


if __name__ == "__main__":
    us_model = MarsStockPredictor('us', r'E:\saved_models\mars_prediction_model\us')
    japan_model = MarsStockPredictor('japan', r'E:\saved_models\mars_prediction_model\japan')
    german_model = MarsStockPredictor('german', r'E:\saved_models\mars_prediction_model\german')

    if us_model.model is None:
        us_model.train()

    if japan_model.model is None:
        japan_model.train()

    if german_model.model is None:
        german_model.train()

    us_model.predict('AAPL')
    japan_model.predict('7203.T')  # toyota
    german_model.predict('SAP.DE')
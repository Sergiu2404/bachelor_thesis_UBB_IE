import numpy as np
import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class StockPricePredictor:
    def __init__(self, look_back=60):
        """
        Initializes the StockPricePredictor with a specified look-back period.
        :param look_back: Number of previous days used for prediction.
        """
        self.look_back = look_back
        self.model = self.create_lstm_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL').
        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format.
        :return: DataFrame with stock price data.
        """
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df

    def prepare_data(self, data):
        """
        Prepares stock price data for LSTM model training.
        :param data: DataFrame containing stock price data.
        :return: Tuple (X, y, scaler) where X is input sequences, y is target values.
        """
        prices = data['Close'].values.reshape(-1, 1)
        prices_scaled = self.scaler.fit_transform(prices)

        X, y = [], []
        for i in range(self.look_back, len(prices_scaled)):
            X.append(prices_scaled[i - self.look_back:i, 0])
            y.append(prices_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def create_lstm_model(self):
        """
        Creates and compiles an LSTM model for stock price prediction.
        :return: Compiled LSTM model.
        """
        model = models.Sequential([
            layers.LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)),
            layers.Dropout(0.2),
            layers.LSTM(units=50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_future(self, last_sequence, num_days):
        """
        Predict future stock prices based on trained LSTM model.
        :param last_sequence: The last known sequence of stock prices.
        :param num_days: Number of days to predict.
        :return: List of predicted stock prices for the next num_days.
        """
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(num_days):
            current_sequence_reshaped = current_sequence.reshape(1, -1, 1)
            next_day_scaled = self.model.predict(current_sequence_reshaped)

            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_day_scaled

            future_predictions.append(next_day_scaled[0])

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        return future_predictions.flatten()

    def run_model(self, ticker, start_date, end_date, num_days_to_predict, epochs=50, batch_size=32):
        """
        Executes the complete stock price prediction workflow.
        :param ticker: Stock ticker symbol.
        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format.
        :param num_days_to_predict: Number of days to predict into the future.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :return: List of tuples (date, predicted_price).
        """
        print(f"Fetching data for {ticker}...")
        df = self.fetch_stock_data(ticker, start_date, end_date)
        X, y = self.prepare_data(df)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print("Training LSTM model...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

        last_sequence = X[-1]
        future_predictions = self.predict_future(last_sequence, num_days_to_predict)

        last_date = datetime.strptime(end_date, '%Y-%m-%d')
        future_dates = [last_date + timedelta(days=x + 1) for x in range(num_days_to_predict)]

        predictions = [(date.date(), price) for date, price in zip(future_dates, future_predictions)]

        print("\nPredicted prices for the next", num_days_to_predict, "days:")
        for date, price in predictions:
            print(f"{date}: ${price:.2f}")

        return predictions


# Testing the model
# if __name__ == "__main__":
#     predictor = StockPricePredictor()
#     predictions = predictor.run_model('AAPL', '2024-01-01', '2024-05-05', 10)
#     print("Test Predictions:", predictions)

# import numpy as np
# import pandas as pd
# import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from fredapi import Fred



class EnhancedStockPricePredictor:
    def __init__(self, look_back_months=36, fred_api_key="15a4d2e1121e1a09cc3021690a867b13"):
        self.look_back_months = look_back_months
        self.feature_count = None  # Will be set during data preparation
        self.monthly_model = None  # Will be created after feature count is known
        self.macro_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.fred_api = Fred(fred_api_key)

    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch monthly stock data"""
        stock = yf.Ticker(ticker)
        # Get daily data then resample to monthly
        df = stock.history(start=start_date, end=end_date, interval="1d")
        # Calculate monthly metrics - using 'ME' instead of 'M'
        monthly_df = df['Close'].resample('ME').agg(['first', 'max', 'min', 'last'])
        monthly_df.columns = ['Open', 'High', 'Low', 'Close']
        monthly_df['Volume'] = df['Volume'].resample('ME').sum()

        # Add technical indicators
        monthly_df['MA_3'] = monthly_df['Close'].rolling(window=3).mean()
        monthly_df['MA_6'] = monthly_df['Close'].rolling(window=6).mean()
        monthly_df['MA_12'] = monthly_df['Close'].rolling(window=12).mean()

        # Calculate returns
        monthly_df['Returns'] = monthly_df['Close'].pct_change()
        monthly_df['Volatility'] = monthly_df['Returns'].rolling(window=3).std()

        return monthly_df

    def fetch_macroeconomic_data(self, start_date, end_date):
        """Fetch monthly macroeconomic indicators from FRED"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Fetch data
        gdp = self.fred_api.get_series("GDP", start_date, end_date)
        gdp = gdp.reindex(pd.date_range(start=gdp.index[0], end=gdp.index[-1], freq='ME'))
        gdp = gdp.ffill()

        cpi = self.fred_api.get_series("CPIAUCSL", start_date, end_date)
        unemployment = self.fred_api.get_series("UNRATE", start_date, end_date)
        interest_rate = self.fred_api.get_series("DGS10", start_date, end_date)
        inflation_expectations = self.fred_api.get_series("T10YIE", start_date, end_date)

        # S&P 500 as a market benchmark - extract just the Close column values
        sp500_data = yf.download("^GSPC", start=start_date, end=end_date, interval="1mo")
        sp500 = sp500_data['Close']

        # Create a properly sized date range for the data frame
        date_range = pd.date_range(start=start_dt, periods=len(sp500), freq='ME')

        #bound them to 60 (expected length for each)
        gdp = gdp.reindex(date_range, method='ffill')
        interest_rate = interest_rate.resample('ME').last()  # Keep last value of each month
        interest_rate = interest_rate.reindex(date_range, method='ffill')  # Ensure 60 values

        inflation_expectations = inflation_expectations.resample('ME').last()
        inflation_expectations = inflation_expectations.reindex(date_range, method='ffill')

        print(len(gdp), len(cpi), len(unemployment), len(interest_rate), len(inflation_expectations), len(sp500),
              len(date_range))

        # macro_data = pd.DataFrame({
        #     'GDP': gdp.values if isinstance(gdp, pd.Series) else gdp,
        #     'CPI': cpi.values if isinstance(cpi, pd.Series) else cpi,
        #     'Unemployment': unemployment.values if isinstance(unemployment, pd.Series) else unemployment,
        #     'Interest_Rate': interest_rate.values if isinstance(interest_rate, pd.Series) else interest_rate,
        #     'Inflation_Expectations': inflation_expectations.values if isinstance(inflation_expectations,
        #                                                                           pd.Series) else inflation_expectations,
        #     'SP500': sp500.values  # Series values are already 1D
        # }, index=date_range)
        macro_data = pd.DataFrame({
            'GDP': gdp.values.squeeze(),  # Ensure 1D
            'CPI': cpi.values.squeeze(),
            'Unemployment': unemployment.values.squeeze(),
            'Interest_Rate': interest_rate.values.squeeze(),
            'Inflation_Expectations': inflation_expectations.values.squeeze(),
            'SP500': sp500.values.squeeze()
        }, index=date_range)

        for col in macro_data.columns:
            # macro_data[f'{col}_Change'] = macro_data[col].pct_change(fill_method=None)
            # macro_data[f'{col}_Change'].fillna(inplace=True)  # Fill forward
            # macro_data[f'{col}_Change'].fillna(inplace=True)  # Fill backward

            macro_data[f'{col}_Change'] = macro_data[col].pct_change()

        macro_data = macro_data.ffill().bfill()
        #macro_data = macro_data.fillna(method='ffill').fillna(method='bfill')

        if macro_data.isna().sum().sum() > 0:
            print("Warning: NaN values detected in macro_data, filling with forward/backward fill.")
            macro_data = macro_data.ffill().bfill()

        return macro_data

    def prepare_lstm_data(self, stock_data, macro_data):
        """Prepare data for LSTM model with merged stock and macro data"""
        stock_data.index = stock_data.index.tz_localize(None)
        macro_data.index = macro_data.index.tz_localize(None)

        merged_data = pd.merge(
            stock_data,
            macro_data,
            left_index=True,
            right_index=True,
            how='inner'
        )

        feature_cols = [col for col in merged_data.columns if col != 'Close']
        self.feature_count = len(feature_cols)

        # create model with the known feature count
        self.monthly_model = self.create_lstm_model()

        # Scale all features
        features_scaled = self.feature_scaler.fit_transform(merged_data[feature_cols])

        # Scale target (Close price)
        close_prices = merged_data['Close'].values.reshape(-1, 1)
        close_scaled = self.price_scaler.fit_transform(close_prices)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.look_back_months, len(merged_data)):
            # For each month, include the previous look_back_months
            X.append(features_scaled[i - self.look_back_months:i])
            y.append(close_scaled[i])

        X, y = np.array(X), np.array(y)

        return X, y, merged_data

    def create_lstm_model(self):
        """Create an LSTM model for monthly stock prediction with correct input shape"""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.look_back_months, self.feature_count)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_macro_adjustment_model(self, stock_data, macro_data, prediction_window=12):
        """Train a secondary model to adjust predictions based on macroeconomic trends"""
        # Prepare data for the adjustment model
        merged_data = pd.merge(
            stock_data,
            macro_data,
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Create features for price movements based on macro factors
        # For each month, calculate the price movement X months ahead
        features = []
        targets = []

        for i in range(len(merged_data) - prediction_window):
            # Current macro conditions and stock features
            current_features = merged_data.iloc[i].drop('Close')

            # Future price change (percentage) after prediction_window months
            future_price = merged_data.iloc[i + prediction_window]['Close']
            current_price = merged_data.iloc[i]['Close']
            price_change_pct = (future_price - current_price) / current_price

            features.append(current_features)
            targets.append(price_change_pct)

        # Train the adjustment model
        X = pd.DataFrame(features)
        y = np.array(targets)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.macro_model.fit(X_train, y_train)

        return self.macro_model

    def predict_future_monthly(self, stock_data, macro_data, num_months=12):
        """Predict stock prices for the next num_months"""
        # Get the latest data for prediction
        merged_data = pd.merge(
            stock_data,
            macro_data,
            left_index=True,
            right_index=True,
            how='inner'
        )

        feature_cols = [col for col in merged_data.columns if col != 'Close']
        features_scaled = self.feature_scaler.transform(merged_data[feature_cols].tail(self.look_back_months))

        # Start with the last sequence of data
        current_sequence = features_scaled[-self.look_back_months:]

        # Future dates for predictions
        last_date = merged_data.index[-1]
        future_dates = [last_date + relativedelta(months=i + 1) for i in range(num_months)]
        future_predictions = []

        # First-pass predictions with LSTM
        for month in range(num_months):
            # Reshape for LSTM
            current_sequence_reshaped = current_sequence.reshape(1, self.look_back_months, -1)

            # Predict next month
            next_month_scaled = self.monthly_model.predict(current_sequence_reshaped, verbose=0)
            next_month_price = self.price_scaler.inverse_transform(next_month_scaled)[0][0]

            future_predictions.append(next_month_price)

            # Simulate next month's features for next prediction
            # This is a simplified approach - in reality we would need forecasts for macro indicators
            next_features = current_sequence[-1].copy()

            # Update the sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_features

        # Apply macroeconomic adjustments to the predictions
        adjusted_predictions = self.apply_macro_adjustments(future_predictions, merged_data, future_dates)

        return list(zip(future_dates, adjusted_predictions))

    def apply_macro_adjustments(self, base_predictions, historical_data, future_dates):
        """Apply adjustments to predictions based on macroeconomic forecasts"""
        recent_macro = historical_data.iloc[-6:].drop('Close', axis=1)  # Last 6 months

        avg_trends = {}
        for col in recent_macro.columns:
            if col.endswith('_Change'):
                avg_trends[col] = recent_macro[col].mean()

        # Apply adjustments
        adjusted_predictions = []
        base_price = historical_data['Close'].iloc[-1]
        epsilon = 1e-6  # Small constant to prevent division by zero

        for i, base_pred in enumerate(base_predictions):
            month = i + 1

            # Base prediction
            if i == 0:
                pred_pct_change = (base_pred - base_price) / base_price
            else:
                safe_last_value = adjusted_predictions[-1] if adjusted_predictions[-1] != 0 else epsilon
                pred_pct_change = (base_pred - safe_last_value) / safe_last_value

            # Stronger adjustments for longer-term predictions
            inflation_factor = 1 + (avg_trends.get('Inflation_Expectations_Change', 0) * month * 0.1)
            interest_factor = 1 - (avg_trends.get('Interest_Rate_Change', 0) * month * 0.15)
            gdp_factor = 1 + (avg_trends.get('GDP_Change', 0) * month * 0.2)

            adjustment = (inflation_factor * interest_factor * gdp_factor - 1)
            adjusted_pct_change = pred_pct_change + adjustment

            if i == 0:
                adjusted_price = base_price * (1 + adjusted_pct_change)
            else:
                adjusted_price = adjusted_predictions[-1] * (1 + adjusted_pct_change)

            adjusted_predictions.append(max(0, adjusted_price))  # Prevent negative prices

        return adjusted_predictions

    def run_model(self, ticker, training_years=5, num_months_to_predict=12, epochs=50, batch_size=32):
        """Run the complete model pipeline"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - relativedelta(years=training_years)).strftime('%Y-%m-%d')

        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        macro_data = self.fetch_macroeconomic_data(start_date, end_date)

        X, y, merged_data = self.prepare_lstm_data(stock_data, macro_data)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print("Training LSTM model...")
        self.monthly_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        print("Training macroeconomic adjustment model...")
        self.train_macro_adjustment_model(stock_data, macro_data)

        print(f"Generating predictions for {ticker} for the next {num_months_to_predict} months...")
        predictions = self.predict_future_monthly(stock_data, macro_data, num_months_to_predict)

        print("\nPredicted monthly prices:")
        for date, price in predictions:
            print(f"{date.strftime('%Y-%m')}: ${price:.2f}")

        # Return simplified monthly prediction values only
        monthly_predictions = [price for _, price in predictions]
        return monthly_predictions


# Example usage
if __name__ == "__main__":
    predictor = EnhancedStockPricePredictor(look_back_months=36)
    predictions = predictor.run_model(
        ticker="AAPL",
        training_years=5,
        num_months_to_predict=12,
        epochs=50
    )

    # Print simplified monthly prediction values
    print("\nSimplified Monthly Predictions for next 12 months:")
    for i, price in enumerate(predictions):
        print(f"Month {i + 1}: ${price:.2f}")
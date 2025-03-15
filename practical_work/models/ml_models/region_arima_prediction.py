import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import random


class RegionalARIMAStockPredictionModel:
    def __init__(self):
        self.ticker = None
        self.model = None
        self.data = None
        self.region = None
        self.base_model_path = "E:\\saved_models\\arima_price_prediction_model"

        self.regions = {
            'us': {
                'path': os.path.join(self.base_model_path, "us"),
                'model_file': "usa_arima_prediction.pkl",
                'benchmark': "^GSPC"
            },
            'german': {
                'path': os.path.join(self.base_model_path, "german"),
                'model_file': "german_arima_prediction.pkl",
                'benchmark': "^GDAXI"
            },
            'japan': {
                'path': os.path.join(self.base_model_path, "japan"),
                'model_file': "japan_arima_prediction.pkl",
                'benchmark': "^N225"
            }
        }

        self.ticker_region_map = {
            '': 'us',  # us stocks typically don't have any prefix

            '.DE': 'german',
            '.F': 'german',
            '.SG': 'german',
            '.BE': 'german',

            '.T': 'japan',
        }

    def set_ticker(self, ticker):
        self.ticker = ticker
        self.region = self.determine_region(ticker)
        return self

    def determine_region(self, ticker):
        for suffix, region in self.ticker_region_map.items():
            if ticker.endswith(suffix):
                return region

        if ticker.isdigit() and len(ticker) == 4:
            return 'japan'
        elif ticker.endswith('.DE') or ticker.endswith('.F'):
            return 'german'
        else:
            return 'us'

    def get_model_path(self):
        if not self.region:
            raise ValueError("Region not determined. Set ticker first.")

        region_info = self.regions.get(self.region)
        if not region_info:
            raise ValueError(f"Unknown region: {self.region}")

        return os.path.join(region_info['path'], region_info['model_file'])

    def get_benchmark_index(self):
        if not self.region:
            raise ValueError("Region not determined. Set ticker first.")

        return self.regions[self.region]['benchmark']

    def fetch_stock_data(self, ticker=None, train_start='2010-01-01', train_end='2020-12-31',
                         val_start='2021-01-01', val_end='2023-01-01'):
        if ticker is None:
            ticker = self.ticker

        if ticker is None:
            raise ValueError("Ticker symbol not set.")

        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=train_start, end=val_end)
        data = data.asfreq('B')
        data = data.ffill().bfill()
        training_data = data['Close'][train_start:train_end]
        validation_data = data['Close'][val_start:val_end]

        return (ticker, data, training_data, validation_data)

    def build_and_train_model(self, training_data, validation_data):
        print("Finding best ARIMA parameters with auto_arima")
        self.model = auto_arima(training_data, seasonal=False, trace=True,
                                error_action='ignore', suppress_warnings=True)
        predictions = self.model.predict(n_periods=len(validation_data))
        mse = mean_squared_error(validation_data, predictions)
        rmse = np.sqrt(mse)
        print(f"Validation RMSE: {rmse:.2f}")
        return predictions

    def predict_next_12_months(self):
        if self.ticker is None:
            raise ValueError("Ticker symbol not set. Use set_ticker() method first.")
        if self.model is None or self.data is None:
            raise ValueError("Model has not been trained or data is not loaded")

        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=252, freq='B')
        predictions = self.model.predict(n_periods=252)

        if isinstance(self.data.iloc[-1], pd.Series):
            last_known_price = float(self.data.iloc[-1].iloc[0])
        else:
            last_known_price = float(self.data.iloc[-1])

        threshold = last_known_price * 0.1

        for i in range(1, len(predictions)):
            current_value = predictions[i - 1]

            if predictions[i] > current_value:
                predictions[i] *= 1.01
            else:
                if current_value < threshold:
                    decrease_factor = 0.99 + 0.01 * (1 - current_value / threshold)
                    decrease_factor = min(decrease_factor, 0.999)
                    predictions[i] = current_value * decrease_factor
                else:
                    predictions[i] *= 0.99

        predictions = np.maximum(predictions, 0)

        future_predictions = pd.Series(predictions, index=future_dates)

        monthly_indices = pd.date_range(start=future_dates.min(), end=future_dates.max(), freq='ME')
        monthly_predictions = [future_predictions.loc[future_predictions.index[
            future_predictions.index.get_indexer([date], method='nearest')[0]]] for date in monthly_indices]

        return monthly_predictions, monthly_indices

    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_path = self.get_model_path()
        dir_path = os.path.dirname(model_path)

        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self):
        model_path = self.get_model_path()

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"No existing model found at {model_path}")
            return False

    def initialize_regional_model(self):
        benchmark_index = self.get_benchmark_index()
        model_path = self.get_model_path()

        if self.load_model():
            print(f"Loaded existing {self.region} model")
            return True

        print(f"No existing {self.region} model found. Training on {benchmark_index}...")
        benchmark_data = self.fetch_stock_data(benchmark_index)

        training_data = benchmark_data[2]
        validation_data = benchmark_data[3]

        self.build_and_train_model(training_data, validation_data)

        self.save_model()
        return True

    def run_stock_prediction(self, ticker_data=None):
        """Run stock prediction using the appropriate regional model"""
        if ticker_data:
            ticker = ticker_data[0]
            self.data = ticker_data[1]
            training_data = ticker_data[2]
            validation_data = ticker_data[3]

            self.set_ticker(ticker)

        if self.ticker is None:
            raise ValueError("Ticker symbol not set")

        print(f"Processing {self.ticker} stock prediction using {self.region} model")

        self.initialize_regional_model()

        all_data = self.data['Close'].asfreq('B').ffill().bfill()
        self.model.update(all_data)

        print("Predicting prices for the next 12 months...")
        predictions, future_dates = self.predict_next_12_months()

        return predictions, future_dates, self.region

    def print_monthly_predictions(self, predictions, future_dates):
        if self.ticker is None:
            raise ValueError("Ticker symbol not set. Use set_ticker() method first.")

        print(f"\n{self.ticker} Monthly Price Predictions (using {self.region} model):")
        print("-" * 60)
        print(f"{'Month':<15} {'Predicted Price':<15}")
        print("-" * 60)
        for i in range(len(predictions)):
            month_str = future_dates[i].strftime('%b %Y')
            price_str = f"${predictions[i]:.2f}"
            print(f"{month_str:<15} {price_str:<15}")


def main():
    model = RegionalARIMAStockPredictionModel()

    us_ticker = 'AAPL'
    german_ticker = 'BMW.DE'
    japanese_ticker = '7203.T'  # toyota

    print("\nUS")
    data_us = model.fetch_stock_data(us_ticker)
    predictions_us, dates_us, region_us = model.run_stock_prediction(data_us)
    model.print_monthly_predictions(predictions_us, dates_us)

    print("\nGerman")
    data_german = model.fetch_stock_data(german_ticker)
    predictions_german, dates_german, region_german = model.run_stock_prediction(data_german)
    model.print_monthly_predictions(predictions_german, dates_german)

    print("\nJapan")
    data_japanese = model.fetch_stock_data(japanese_ticker)
    predictions_japanese, dates_japanese, region_japanese = model.run_stock_prediction(data_japanese)
    model.print_monthly_predictions(predictions_japanese, dates_japanese)


if __name__ == "__main__":
    main()
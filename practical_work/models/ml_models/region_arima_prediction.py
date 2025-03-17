import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import random


class ARIMAStockPredictor:
    def __init__(self, region, model_path):
        self.region = region
        self.model_path = model_path
        self.model_file = os.path.join(model_path, f"{region}_arima_model.joblib")
        self.model_params_file = os.path.join(model_path, f"{region}_arima_params.joblib")
        self.model = None
        self.model_params = None
        self.training_data = None

        self.region_config = {
            'us': {'index': '^GSPC', 'name': 'S&P 500'},
            'japan': {'index': '^N225', 'name': 'Nikkei 225'},
            'german': {'index': '^GDAXI', 'name': 'DAX'}
        }

        os.makedirs(model_path, exist_ok=True)

        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.model_params_file):
                print(f"Loading existing model for {self.region} region...")
                self.model = joblib.load(self.model_file)
                self.model_params = joblib.load(self.model_params_file)
                print("Model loaded successfully!")
                return True
            else:
                print(f"No existing model found for {self.region} region.")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _transform_to_returns(self, prices):
        if isinstance(prices, pd.Series):
            returns = np.log(prices).diff().dropna()
            return returns
        else:
            log_prices = np.log(prices)
            returns = np.diff(log_prices)
            return returns

    def _transform_to_prices(self, returns, last_price):
        cumulative_returns = np.cumsum(returns)
        return last_price * np.exp(cumulative_returns)

    def train(self, years=5, order=(2, 1, 2), seasonal_order=(1, 0, 1, 5)):
        if self.region not in self.region_config:
            raise ValueError(f"Invalid region: {self.region}")

        index_symbol = self.region_config[self.region]['index']
        index_name = self.region_config[self.region]['name']

        print(f"Training model for {self.region} region using {index_name} ({index_symbol})...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        df = yf.download(index_symbol, start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"Could not download data for {index_symbol}")

        print(f"Downloaded {len(df)} days of historical data for {index_name}")

        prices = df['Close']
        returns = self._transform_to_returns(prices)

        # SARIMAX for better modeling of financial time series
        model = SARIMAX(
            returns,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fit_model = model.fit(disp=False)

        # Save model and parameters
        joblib.dump(fit_model, self.model_file)
        joblib.dump({
            'order': order,
            'seasonal_order': seasonal_order,
            'last_price': float(prices.iloc[-1]),
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'train_end_date': end_date
        }, self.model_params_file)

        self.model = fit_model
        self.model_params = {
            'order': order,
            'seasonal_order': seasonal_order,
            'last_price': float(prices.iloc[-1]),
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'train_end_date': end_date
        }

        print(f"\nModel training complete for {self.region} region")

        return fit_model

    def predict(self, ticker, forecast_periods=252):
        if self.model is None or self.model_params is None:
            raise ValueError("Model not trained or loaded. Call train() first.")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)

        print(f"Downloading data for {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            raise ValueError(f"Could not download data for {ticker}")

        current_price = float(stock_data['Close'].iloc[-1])

        if len(stock_data) > 30:
            stock_returns = stock_data['Close'].pct_change().dropna()

            market_data = yf.download(self.region_config[self.region]['index'], start=start_date, end=end_date)
            market_returns = market_data['Close'].pct_change().dropna()

            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            aligned_data.columns = ['stock', 'market']

            cov_matrix = aligned_data.cov()
            beta = cov_matrix.loc['stock', 'market'] / cov_matrix.loc['market', 'market']

            beta = max(0.5, min(2.0, beta))
        else:
            beta = 1.0

        forecast_result = self.model.forecast(steps=forecast_periods)

        if isinstance(forecast_result, pd.Series):
            forecast_result = forecast_result.values

        market_volatility = self.model_params['std_return']
        forecast_with_noise = np.array([])

        prev_return = forecast_result[0]
        forecast_with_noise = np.append(forecast_with_noise, prev_return)

        for i in range(1, forecast_periods):
            reversion_strength = 0.05
            mean_reversion = reversion_strength * (self.model_params['mean_return'] - prev_return)

            volatility = market_volatility * beta
            random_component = np.random.normal(0, volatility)

            weight_forecast = max(0.8 - (i / forecast_periods), 0.1)
            weight_reversion = 0.2
            weight_random = 1.0 - weight_forecast - weight_reversion

            forecast_idx = min(i, len(forecast_result) - 1)
            new_return = (
                    weight_forecast * forecast_result[forecast_idx] +
                    weight_reversion * mean_reversion +
                    weight_random * random_component
            )

            forecast_with_noise = np.append(forecast_with_noise, new_return)
            prev_return = new_return

        forecast_prices = self._transform_to_prices(forecast_with_noise, current_price)

        last_date = stock_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='B')

        forecast_df = pd.DataFrame({
            'Date': forecast_dates[:len(forecast_prices)],
            'Forecasted_Price': forecast_prices
        })

        twelve_month_idx = min(252 - 1, len(forecast_prices) - 1)
        predicted_price_12m = float(forecast_prices[twelve_month_idx])
        predicted_return = (predicted_price_12m / current_price) - 1

        if predicted_return > 1.0:  # more than 100% return
            predicted_return = min(predicted_return, 0.30)  # cap at 30%
            predicted_price_12m = current_price * (1 + predicted_return)
        elif predicted_return < -0.5:  # more than 50% loss
            predicted_return = max(predicted_return, -0.30)
            predicted_price_12m = current_price * (1 + predicted_return)

        print(f"\nPrediction for {ticker} using {self.region} model:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted 12-Month Return: {predicted_return * 100:.2f}%")
        print(f"Predicted Price in 12 Months: ${predicted_price_12m:.2f}")
        print(f"Stock Beta: {beta:.2f}")

        step = 21  # ~monthly
        if len(forecast_df) >= step:
            monthly_indices = list(range(0, len(forecast_df), step))
            monthly_forecast = forecast_df.iloc[monthly_indices]
            print("\nMonthly forecast:")
            for _, row in monthly_forecast.iterrows():
                print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Forecasted_Price']:.2f}")

        return predicted_return, current_price, predicted_price_12m, forecast_df

    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() first.")

        return self.model.summary()


if __name__ == "__main__":
    np.random.seed(42)

    us_model = ARIMAStockPredictor('us', r'E:\saved_models\arima_price_prediction_model\us')
    japan_model = ARIMAStockPredictor('japan', r'E:\saved_models\arima_price_prediction_model\japan')
    german_model = ARIMAStockPredictor('german', r'E:\saved_models\arima_price_prediction_model\german')

    if us_model.model is None:
        us_model.train()

    if japan_model.model is None:
        japan_model.train()

    if german_model.model is None:
        german_model.train()

    us_model.predict('AAPL')
    japan_model.predict('7203.T')
    german_model.predict('SAP.DE')





# import os
# import joblib
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from datetime import datetime, timedelta
# from pmdarima import auto_arima
# from sklearn.metrics import mean_squared_error
# import random
#
#
# class RegionalARIMAStockPredictionModel:
#     def __init__(self):
#         self.ticker = None
#         self.model = None
#         self.data = None
#         self.region = None
#         self.base_model_path = "E:\\saved_models\\arima_price_prediction_model"
#
#         self.regions = {
#             'us': {
#                 'path': os.path.join(self.base_model_path, "us"),
#                 'model_file': "usa_arima_prediction.pkl",
#                 'benchmark': "^GSPC"
#             },
#             'german': {
#                 'path': os.path.join(self.base_model_path, "german"),
#                 'model_file': "german_arima_prediction.pkl",
#                 'benchmark': "^GDAXI"
#             },
#             'japan': {
#                 'path': os.path.join(self.base_model_path, "japan"),
#                 'model_file': "japan_arima_prediction.pkl",
#                 'benchmark': "^N225"
#             }
#         }
#
#         self.ticker_region_map = {
#             '': 'us',  # us stocks typically don't have any prefix
#
#             '.DE': 'german',
#             '.F': 'german',
#             '.SG': 'german',
#             '.BE': 'german',
#
#             '.T': 'japan',
#         }
#
#     def set_ticker(self, ticker):
#         self.ticker = ticker
#         self.region = self.determine_region(ticker)
#         return self
#
#     def determine_region(self, ticker):
#         for suffix, region in self.ticker_region_map.items():
#             if ticker.endswith(suffix):
#                 return region
#
#         if ticker.isdigit() and len(ticker) == 4:
#             return 'japan'
#         elif ticker.endswith('.DE') or ticker.endswith('.F'):
#             return 'german'
#         else:
#             return 'us'
#
#     def get_model_path(self):
#         if not self.region:
#             raise ValueError("Region not determined. Set ticker first.")
#
#         region_info = self.regions.get(self.region)
#         if not region_info:
#             raise ValueError(f"Unknown region: {self.region}")
#
#         return os.path.join(region_info['path'], region_info['model_file'])
#
#     def get_benchmark_index(self):
#         if not self.region:
#             raise ValueError("Region not determined. Set ticker first.")
#
#         return self.regions[self.region]['benchmark']
#
#     def fetch_stock_data(self, ticker=None, train_start='2010-01-01', train_end='2020-12-31',
#                          val_start='2021-01-01', val_end='2023-01-01'):
#         if ticker is None:
#             ticker = self.ticker
#
#         if ticker is None:
#             raise ValueError("Ticker symbol not set.")
#
#         print(f"Downloading data for {ticker}...")
#         data = yf.download(ticker, start=train_start, end=val_end)
#         data = data.asfreq('B')
#         data = data.ffill().bfill()
#         training_data = data['Close'][train_start:train_end]
#         validation_data = data['Close'][val_start:val_end]
#
#         return (ticker, data, training_data, validation_data)
#
#     def build_and_train_model(self, training_data, validation_data):
#         print("Finding best ARIMA parameters with auto_arima")
#         self.model = auto_arima(training_data, seasonal=False, trace=True,
#                                 error_action='ignore', suppress_warnings=True)
#         predictions = self.model.predict(n_periods=len(validation_data))
#         mse = mean_squared_error(validation_data, predictions)
#         rmse = np.sqrt(mse)
#         print(f"Validation RMSE: {rmse:.2f}")
#         return predictions
#
#     def predict_next_12_months(self):
#         if self.ticker is None:
#             raise ValueError("Ticker symbol not set. Use set_ticker() method first.")
#         if self.model is None or self.data is None:
#             raise ValueError("Model has not been trained or data is not loaded")
#
#         last_date = self.data.index[-1]
#         future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
#                                      periods=252, freq='B')
#         predictions = self.model.predict(n_periods=252)
#
#         if isinstance(self.data.iloc[-1], pd.Series):
#             last_known_price = float(self.data.iloc[-1].iloc[0])
#         else:
#             last_known_price = float(self.data.iloc[-1])
#
#         threshold = last_known_price * 0.1
#
#         for i in range(1, len(predictions)):
#             current_value = predictions[i - 1]
#
#             if predictions[i] > current_value:
#                 predictions[i] *= 1.01
#             else:
#                 if current_value < threshold:
#                     decrease_factor = 0.99 + 0.01 * (1 - current_value / threshold)
#                     decrease_factor = min(decrease_factor, 0.999)
#                     predictions[i] = current_value * decrease_factor
#                 else:
#                     predictions[i] *= 0.99
#
#         predictions = np.maximum(predictions, 0)
#
#         future_predictions = pd.Series(predictions, index=future_dates)
#
#         monthly_indices = pd.date_range(start=future_dates.min(), end=future_dates.max(), freq='ME')
#         monthly_predictions = [future_predictions.loc[future_predictions.index[
#             future_predictions.index.get_indexer([date], method='nearest')[0]]] for date in monthly_indices]
#
#         return monthly_predictions, monthly_indices
#
#     def save_model(self):
#         if self.model is None:
#             raise ValueError("No model to save. Train a model first.")
#
#         model_path = self.get_model_path()
#         dir_path = os.path.dirname(model_path)
#
#         os.makedirs(dir_path, exist_ok=True)
#         joblib.dump(self.model, model_path)
#         print(f"Model saved to {model_path}")
#
#     def load_model(self):
#         model_path = self.get_model_path()
#
#         if os.path.exists(model_path):
#             self.model = joblib.load(model_path)
#             print(f"Model loaded from {model_path}")
#             return True
#         else:
#             print(f"No existing model found at {model_path}")
#             return False
#
#     def initialize_regional_model(self):
#         benchmark_index = self.get_benchmark_index()
#         model_path = self.get_model_path()
#
#         if self.load_model():
#             print(f"Loaded existing {self.region} model")
#             return True
#
#         print(f"No existing {self.region} model found. Training on {benchmark_index}...")
#         benchmark_data = self.fetch_stock_data(benchmark_index)
#
#         training_data = benchmark_data[2]
#         validation_data = benchmark_data[3]
#
#         self.build_and_train_model(training_data, validation_data)
#
#         self.save_model()
#         return True
#
#     def run_stock_prediction(self, ticker_data=None):
#         """Run stock prediction using the appropriate regional model"""
#         if ticker_data:
#             ticker = ticker_data[0]
#             self.data = ticker_data[1]
#             training_data = ticker_data[2]
#             validation_data = ticker_data[3]
#
#             self.set_ticker(ticker)
#
#         if self.ticker is None:
#             raise ValueError("Ticker symbol not set")
#
#         print(f"Processing {self.ticker} stock prediction using {self.region} model")
#
#         self.initialize_regional_model()
#
#         all_data = self.data['Close'].asfreq('B').ffill().bfill()
#         self.model.update(all_data)
#
#         print("Predicting prices for the next 12 months...")
#         predictions, future_dates = self.predict_next_12_months()
#
#         return predictions, future_dates, self.region
#
#     def print_monthly_predictions(self, predictions, future_dates):
#         if self.ticker is None:
#             raise ValueError("Ticker symbol not set. Use set_ticker() method first.")
#
#         print(f"\n{self.ticker} Monthly Price Predictions (using {self.region} model):")
#         print("-" * 60)
#         print(f"{'Month':<15} {'Predicted Price':<15}")
#         print("-" * 60)
#         for i in range(len(predictions)):
#             month_str = future_dates[i].strftime('%b %Y')
#             price_str = f"${predictions[i]:.2f}"
#             print(f"{month_str:<15} {price_str:<15}")
#
#
# def main():
#     model = RegionalARIMAStockPredictionModel()
#
#     us_ticker = 'AAPL'
#     german_ticker = 'BMW.DE'
#     japanese_ticker = '7203.T'  # toyota
#
#     print("\nUS")
#     data_us = model.fetch_stock_data(us_ticker)
#     predictions_us, dates_us, region_us = model.run_stock_prediction(data_us)
#     model.print_monthly_predictions(predictions_us, dates_us)
#
#     print("\nGerman")
#     data_german = model.fetch_stock_data(german_ticker)
#     predictions_german, dates_german, region_german = model.run_stock_prediction(data_german)
#     model.print_monthly_predictions(predictions_german, dates_german)
#
#     print("\nJapan")
#     data_japanese = model.fetch_stock_data(japanese_ticker)
#     predictions_japanese, dates_japanese, region_japanese = model.run_stock_prediction(data_japanese)
#     model.print_monthly_predictions(predictions_japanese, dates_japanese)
#
#
# if __name__ == "__main__":
#     main()
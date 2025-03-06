import random

import numpy as np
import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers
from datetime import datetime, timedelta
import os

class LSTMStockPricePredictor:
    def __init__(self, seq_length=60, model_path='E:\\saved_models\\lstm_prediction_model\\lstm_model.h5'):
        self.seq_length = seq_length
        self.model_path = model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def create_sequences(self, data):
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            X.append(data[i-self.seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def prepare_stock_data(
            self,
            ticker,
            train_start='2010-01-01',
            train_end='2020-12-31',
            val_start='2021-01-01',
            val_end='2023-01-01'
    ):
        data = yf.download(ticker, start=train_start, end=val_end)

        training_data = data['Close'][train_start:train_end]
        validation_data = data['Close'][val_start:val_end]

        training_set = training_data.values.reshape(-1, 1)
        validation_set = validation_data.values.reshape(-1, 1)

        training_set_scaled = self.scaler.fit_transform(training_set)
        validation_set_scaled = self.scaler.transform(validation_set)

        X_train, y_train = self.create_sequences(training_set_scaled)
        X_validation, y_validation = self.create_sequences(validation_set_scaled)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))

        return X_train, y_train, X_validation, y_validation, data

    def build_model(self, input_shape):
        model = models.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=32):
        if os.path.exists(self.model_path):
            print("load existing model")
            self.model = models.load_model(self.model_path)
        else:
            print("build, train, save new model")
            self.model = self.build_model((X_train.shape[1], 1))
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation), verbose=1)
            self.model.save(self.model_path)

    def prepare_latest_data(self, ticker):
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.seq_length * 2)).strftime('%Y-%m-%d')
        latest_data = yf.download(ticker, start=start_date, end=today)

        latest_prices = latest_data['Close'].values.reshape(-1, 1)
        latest_prices_scaled = self.scaler.transform(latest_prices)

        X_latest = np.array([latest_prices_scaled[-self.seq_length:, 0]])
        X_latest = X_latest.reshape((X_latest.shape[0], X_latest.shape[1], 1))

        return X_latest, latest_data

    def predict_next_12_months(self, X_latest):
        current_sequence = X_latest[0]
        predictions = []
        current_date = datetime.now()
        future_dates = []

        for i in range(12):
            future_date = (current_date + timedelta(days=30 * (i + 1)))
            future_dates.append(future_date)

            current_batch = current_sequence[-self.seq_length:].reshape(1, self.seq_length, 1)
            next_pred = self.model.predict(current_batch, verbose=0)
            predictions.append(next_pred[0, 0])

            influence = 0.7 + 0.03 * i  # increase influence of the last prediction
            weighted_pred = (current_sequence[-1] * influence + next_pred[0, 0]) / (1 + influence)
            current_sequence = np.append(current_sequence, weighted_pred)

        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array)

        return predictions_rescaled, future_dates

    def print_monthly_predictions(self, predictions, future_dates, ticker):
        print(f"\n{ticker} Monthly Price Predictions:")
        print(f"{'Month':<15} {'Predicted Price':<15}")
        print("-" * 40)

        for i in range(len(predictions)):
            month_str = future_dates[i].strftime('%b %Y')
            price_str = f"${predictions[i][0]:.2f}"
            print(f"{month_str:<15} {price_str:<15}")

    def run(self, ticker):
        print(f"Loading and preparing {ticker} data...")
        X_train, y_train, X_validation, y_validation, historical_data = self.prepare_stock_data(ticker)
        self.train_model(X_train, y_train, X_validation, y_validation)

        print(f"prepare {ticker} data for pred")
        latest_sequence, latest_data = self.prepare_latest_data(ticker)

        print("predict prices for the next 12 months")
        predictions, future_dates = self.predict_next_12_months(latest_sequence)
        self.print_monthly_predictions(predictions, future_dates, ticker)

# predictor = LSTMStockPricePredictor()
# predictor.run('SPY')












import os
import joblib
from datetime import datetime, timedelta
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error


class ARIMAStockPredictionModel:
    def __init__(self):
        self.ticker = None
        self.model = None
        self.data = None
        self.model_path = "E:\\saved_models\\arima_price_prediction_model"
        self.model_file = os.path.join(self.model_path, "arima_model.pkl")

    def set_ticker(self, ticker):
        """Set the stock ticker symbol to analyze"""
        self.ticker = ticker
        return self

    # def prepare_stock_data(self, train_start='2010-01-01', train_end='2020-12-31',
    #                        val_start='2021-01-01', val_end='2023-01-01'):
    #     if self.ticker is None:
    #         raise ValueError("Ticker symbol not set. Use set_ticker() method first.")
    #
    #     data = yf.download(self.ticker, start=train_start, end=val_end)
    #     data = data.asfreq('B')
    #     # data = data.fillna(method='ffill').fillna(method='bfill')
    #     data = data.ffill().bfill()
    #     training_data = data['Close'][train_start:train_end]
    #     validation_data = data['Close'][val_start:val_end]
    #     self.data = data
    #     return training_data, validation_data

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

        # Introduce a small bias to integrate volatility
        for i in range(1, len(predictions)):
            if predictions[i] > predictions[i - 1]:
                predictions[i] *= 1.01
            else:
                predictions[i] *= 0.99

        future_predictions = pd.Series(predictions, index=future_dates)

        monthly_indices = pd.date_range(start=future_dates.min(), end=future_dates.max(), freq='ME')
        monthly_predictions = [future_predictions.loc[future_predictions.index[
            future_predictions.index.get_indexer([date], method='nearest')[0]]] for date in monthly_indices]

        return monthly_predictions, monthly_indices

    def predict_next_5_years(self):
        if self.ticker is None:
            raise ValueError("Ticker symbol not set. Use set_ticker() method first.")
        if self.model is None or self.data is None:
            raise ValueError("Model has not been trained or data is not loaded")

        last_date = self.data.index[-1]
        # 252 business days / year
        future_periods = 252 * 5
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=future_periods, freq='B')
        predictions = self.model.predict(n_periods=future_periods)

        # introduce a small bias to integrate volatility
        for i in range(1, len(predictions)):
            if predictions[i] > predictions[i - 1]:
                predictions[i] *= 1.001
            else:
                predictions[i] *= 0.999


        future_predictions = pd.Series(predictions, index=future_dates)

        # Generate monthly indices over the 5-year period
        monthly_indices = pd.date_range(start=future_dates.min(), end=future_dates.max(), freq='M')
        monthly_predictions = [future_predictions.loc[future_predictions.index[
            future_predictions.index.get_indexer([date], method='nearest')[0]]] for date in monthly_indices]

        return monthly_predictions, monthly_indices

    def print_monthly_predictions(self, predictions, future_dates):
        if self.ticker is None:
            raise ValueError("Ticker symbol not set. Use set_ticker() method first.")

        print(f"\n{self.ticker} Monthly Price Predictions:")
        print("-" * 40)
        print(f"{'Month':<15} {'Predicted Price':<15}")
        print("-" * 40)
        for i in range(len(predictions)):
            month_str = future_dates[i].strftime('%b %Y')
            price_str = f"${predictions[i]:.2f}"
            print(f"{month_str:<15} {price_str:<15}")

    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.model, self.model_file)
        print(f"Model saved to {self.model_file}")

    def load_model(self):
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            print(f"Model loaded from {self.model_file}")
            return True
        else:
            print(f"No existing model found at {self.model_file}")
            return False

    def run_stock_prediction(self, data):
        ticker = data[0]
        self.data = data[1]

        if ticker is not None:
            self.set_ticker(ticker)

        if self.ticker is None:
            raise ValueError("Ticker symbol not set. Use set_ticker() method or provide ticker parameter.")

        print(f"Processing {self.ticker} stock prediction")

        # Always load fresh data for the current ticker
        print(f"Loading {self.ticker} data")
        #training_data, validation_data = self.prepare_stock_data()

        # Try to load existing model first
        # if self.load_model():
        #     # Update the model with current ticker data
        #     all_data = self.data['Close'].asfreq('B').fillna(method='ffill').fillna(method='bfill')
        #     self.model.update(all_data)
        # else:
        #     # Need to train new model
        #     print("Training ARIMA model")
        #     val_predictions = self.build_and_train_model(training_data, validation_data)
        #
        #     # Refit model on all available data for future predictions
        #     all_data = self.data['Close'].asfreq('B').fillna(method='ffill').fillna(method='bfill')
        #     self.model.update(all_data)
        #
        #     # Save the trained model
        #     self.save_model()
        if self.load_model():
            # Update the model with current ticker data
            all_data = self.data['Close'].asfreq('B').ffill().bfill()
            self.model.update(all_data)
        else:
            # Need to train new model
            training_data = data[2]
            validation_data = data[3]

            print("Training ARIMA model")
            val_predictions = self.build_and_train_model(training_data, validation_data)

            # Refit model on all available data for future predictions
            all_data = self.data['Close'].asfreq('B').ffill().bfill()
            self.model.update(all_data)

            # Save the trained model
            self.save_model()

        print("Predicting prices for the next 12 months...")
        predictions, future_dates = self.predict_next_12_months()
        #predictions, future_dates = self.predict_next_5_years()
        #self.print_monthly_predictions(predictions, future_dates)

        return predictions, future_dates

    def run_stock_prediction_overall_sentiment(self, weighted_sentiment_score, ticker):
        if ticker is not None:
            self.set_ticker(ticker)

        if self.ticker is None:
            raise ValueError("Ticker symbol not set. Use set_ticker() method or provide ticker parameter.")

        print(f"Processing {self.ticker} stock prediction")

        # Always load fresh data for the current ticker
        print(f"Loading {self.ticker} data")
        training_data, validation_data = self.prepare_stock_data()

        if self.load_model():
            # Update the model with current ticker data
            all_data = self.data['Close'].asfreq('B').ffill().bfill()
            self.model.update(all_data)
        else:
            # Need to train new model
            print("Training ARIMA model")
            val_predictions = self.build_and_train_model(training_data, validation_data)

            # Refit model on all available data for future predictions
            all_data = self.data['Close'].asfreq('B').ffill().bfill()
            self.model.update(all_data)

            # Save the trained model
            self.save_model()

        print("Predicting prices for the next 12 months...")
        predictions, future_dates = self.predict_next_12_months()

        # Now adjust the predictions based on the sentiment score
        adjusted_predictions = self.adjust_predictions_with_sentiment(predictions, weighted_sentiment_score)

        return adjusted_predictions, future_dates

    def adjust_predictions_with_sentiment(self, predictions, weighted_sentiment_score):
        adjusted_predictions = []
        score_intervals = {
            (-1, -0.75): (0.15, 0.2),
            (-0.75, -0.2): (0.1, 0.15),
            (-0.2, 0.2): (0.02, 0.1),
            (0.2, 0.75): (0.1, 0.15),
            (0.75, 1): (0.15, 0.2)
        }
        initial_impact_factor = 0.01

        for (low, high), impact in score_intervals.items():
            if low <= weighted_sentiment_score <= high:
                initial_impact_factor = random.uniform(impact[0], impact[1])


        # TODO IDEA: adjust the decay factor (0.6) depending on words in the news article (for words like 'long'/'much time' make it larger: 0.9 and for 'short' do it lower)
        for i, prediction in enumerate(predictions):
            # use exponential decay for a gradual decrease of the impact over the months
            gradual_impact = initial_impact_factor * (0.6 ** i)
            adjusted_prediction = prediction * (1 + weighted_sentiment_score * gradual_impact)
            adjusted_predictions.append(adjusted_prediction)

            # if i == 0:  # apply more impact to the first month
            #     adjusted_prediction = prediction * (1 + weighted_sentiment_score * 0.1)
            # elif i == 1:
            #     adjusted_prediction = prediction * (
            #                 1 + weighted_sentiment_score * 0.03)
            # else:
            #     # rest of months very small effect/impact
            #     adjusted_prediction = prediction * (
            #                 1 + weighted_sentiment_score * 0.01)
            #
            # adjusted_predictions.append(adjusted_prediction)

        return adjusted_predictions

def fetch_stock_data(ticker, train_start='2010-01-01', train_end='2020-12-31',
                       val_start='2021-01-01', val_end='2023-01-01'):
    if ticker is None:
        raise ValueError("Ticker symbol not set. Use set_ticker() method first.")

    data = yf.download(ticker, start=train_start, end=val_end)
    data = data.asfreq('B')
    # data = data.fillna(method='ffill').fillna(method='bfill')
    data = data.ffill().bfill()
    training_data = data['Close'][train_start:train_end]
    validation_data = data['Close'][val_start:val_end]

    return (ticker, data, training_data, validation_data)

def main():
    # Example usage
    model = ARIMAStockPredictionModel()

    data = fetch_stock_data('AAPL')
    print("\nPrediction")
    predictions, dates = model.run_stock_prediction(data)
    for i in range(len(predictions)):
        print(f"price {predictions[i]} at date {dates[i]}")


if __name__ == "__main__":
    main()










# # import matplotlib.pyplot as plt
# # import matplotlib.dates as mdates
# from datetime import datetime, timedelta
# from pmdarima import auto_arima
# from sklearn.metrics import mean_squared_error
#
# def prepare_stock_data(ticker, train_start='2010-01-01', train_end='2020-12-31',
#                       val_start='2021-01-01', val_end='2023-01-01'):
#     data = yf.download(ticker, start=train_start, end=val_end)
#     data = data.asfreq('B')
#     data = data.fillna(method='ffill').fillna(method='bfill')
#     training_data = data['Close'][train_start:train_end]
#     validation_data = data['Close'][val_start:val_end]
#     return training_data, validation_data, data
#
# def predict_next_12_months(model, historical_data):
#     last_date = historical_data.index[-1]
#     future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
#                                  periods=252, freq='B')
#     predictions = model.predict(n_periods=252)
#
#     # Introduce a small bias to integrate volatility
#     for i in range(1, len(predictions)):
#         if predictions[i] > predictions[i-1]:
#             predictions[i] *= 1.01
#         else:
#             predictions[i] *= 0.99
#
#     future_predictions = pd.Series(predictions, index=future_dates)
#
#     monthly_indices = pd.date_range(start=future_dates.min(), end=future_dates.max(), freq='M')
#     monthly_predictions = [future_predictions.loc[future_predictions.index[future_predictions.index.get_indexer([date], method='nearest')[0]]] for date in monthly_indices]
#
#     return monthly_predictions, monthly_indices
#
#
# def build_and_train_model(training_data, validation_data):
#     print("find best ARIMA parameters with auto_arima")
#     model = auto_arima(training_data, seasonal=False, trace=True,
#                        error_action='ignore', suppress_warnings=True)
#     predictions = model.predict(n_periods=len(validation_data))
#     mse = mean_squared_error(validation_data, predictions)
#     rmse = np.sqrt(mse)
#     print(f"Validation RMSE: {rmse:.2f}")
#     return model, predictions
#
# def print_monthly_predictions(predictions, future_dates, ticker):
#     print(f"\n{ticker} Monthly Price Predictions:")
#     print("-" * 40)
#     print(f"{'Month':<15} {'Predicted Price':<15}")
#     print("-" * 40)
#     for i in range(len(predictions)):
#         month_str = future_dates[i].strftime('%b %Y')
#         price_str = f"${predictions[i]:.2f}"
#         print(f"{month_str:<15} {price_str:<15}")
#
# def run_stock_prediction(ticker='AAPL'):
#     print(f"Load {ticker} data")
#     training_data, validation_data, historical_data = prepare_stock_data(ticker)
#
#     print("train auto_arima")
#     model, val_predictions = build_and_train_model(training_data, validation_data)
#
#     print("Refit model on all available data for future predictions")
#     all_data = historical_data['Close'].asfreq('B').fillna(method='ffill').fillna(method='bfill')
#     model.update(all_data)
#
#     print("Predicting prices for the next 12 months...")
#     predictions, future_dates = predict_next_12_months(model, historical_data)
#     print_monthly_predictions(predictions, future_dates, ticker)
#
#     return predictions, future_dates
#
# print("\Prediction for AAPL")
# aapl_predictions, aapl_dates = run_stock_prediction('AAPL')
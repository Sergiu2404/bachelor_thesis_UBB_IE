import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def get_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df


def prepare_data_for_model(df):
    return df['Close']


def train_ets_model(series):
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=5,
        damped_trend=True
    )
    fitted_model = model.fit(optimized=True, use_brute=False)
    return fitted_model


def predict_future_prices(model, days=10):
    # Make predictions
    forecast = model.forecast(days)
    return forecast


def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def main(ticker="AAPL", train_period="6mo", prediction_days=10):
    start_time = time.time()

    df = get_stock_data(ticker, period=train_period)

    price_series = prepare_data_for_model(df)
    model = train_ets_model(price_series)

    forecast = predict_future_prices(model, days=prediction_days)

    execution_time = time.time() - start_time

    last_date = price_series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='B')

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Close': forecast.values
    })

    train_predictions = model.fittedvalues
    train_actual = price_series.values
    evaluation = evaluate_model(train_actual, train_predictions)

    return {
        'model': model,
        'forecast': forecast_df,
        'execution_time': execution_time,
        'evaluation': evaluation
    }


if __name__ == "__main__":
    ticker = "AAPL"
    result = main(ticker)

    print(f"Model trained and predictions made in {result['execution_time']:.2f} seconds")
    print("\nPredicted prices for the next 10 days:")
    print(result['forecast'])
    print("\nModel evaluation on training data:")
    for metric, value in result['evaluation'].items():
        print(f"{metric}: {value:.4f}")
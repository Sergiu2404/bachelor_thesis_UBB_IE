import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta

def predict_stock_arima(ticker, months=3, forecast_days=10):
    end = datetime.today()
    start = end - timedelta(days=months * 30)
    df = yf.download(ticker, start=start, end=end, progress=False)

    close_prices = df['Close'].dropna()
    close_prices.index = pd.to_datetime(close_prices.index)

    if close_prices.empty or close_prices.isna().all():
        raise ValueError("Price data is empty or all NaN")

    model = ARIMA(close_prices, order=(5, 1, 0))  # simple ARIMA(p=5, d=1, q=0)
    fitted = model.fit()

    forecast = fitted.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=close_prices.index[-1] + BDay(), periods=forecast_days, freq=BDay())

    return pd.Series(forecast, index=future_dates)

predicted = predict_stock_arima("AAPL", months=3, forecast_days=10)
print(predicted)

from fake_news_detection import FinancialNewsCredibilityAnalyzer
from news_sentiment import FinancialNewsAnalyzer
from price_predictor import ARIMAStockPredictionModel
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import time
from datetime import datetime, timedelta
import pandas as pd

def get_sentiment_result(news_text):
    sentiment_analyzer = FinancialNewsAnalyzer()
    return sentiment_analyzer.analyze_sentiment(news_text)

def get_credibility_result(news_text):
    credibility_analyzer = FinancialNewsCredibilityAnalyzer()
    return credibility_analyzer.analyze(news_text)['credibility_score']

def split_date_range(start_date, end_date, chunk_days=730):
    date_ranges = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        date_ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return date_ranges

def fetch_stock_data_for_range(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.asfreq('B').ffill().bfill()
    return data

def fetch_stock_data_parallel(ticker, train_start, train_end, val_start, val_end):
    if ticker is None:
        raise ValueError("Ticker symbol not set.")

    start_time = time.time()
    date_ranges = split_date_range(pd.to_datetime(train_start), pd.to_datetime(val_end))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_stock_data_for_range, ticker, start, end) for start, end in date_ranges]
        all_data = [future.result() for future in futures]

    data = pd.concat(all_data).drop_duplicates()
    data = data.asfreq('B').ffill().bfill()
    training_data = data['Close'][train_start:train_end]
    validation_data = data['Close'][val_start:val_end]

    print(f"Stock data fetched in {time.time() - start_time} seconds")
    return (ticker, data, training_data, validation_data)

if __name__ == '__main__':
    news_article = "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"
    ticker = input("Company symbol: ")

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        stock_data_future = executor.submit(
            fetch_stock_data_parallel,
            ticker,
            train_start='2000-01-01',
            train_end='2020-12-31',
            val_start='2021-01-01',
            val_end='2023-01-01'
        )
        credibility_future = executor.submit(get_credibility_result, news_article)
        sentiment_future = executor.submit(get_sentiment_result, news_article)

        stock_data_result = stock_data_future.result()
        sentiment_result = sentiment_future.result()
        credibility_result = credibility_future.result()

    print(sentiment_result['sentiment_score'], credibility_result)

    weighted_credible_sentiment = sentiment_result['sentiment_score'] * credibility_result
    print(f"Weighted sentiment: {weighted_credible_sentiment}")

    prediction_model = ARIMAStockPredictionModel()
    predictions, dates = prediction_model.run_stock_prediction(stock_data_result)

    for i in range(len(predictions)):
        print(f"Price {predictions[i]} at date {dates[i]}")

    print(f"Time elapsed: {time.time() - start_time} seconds")

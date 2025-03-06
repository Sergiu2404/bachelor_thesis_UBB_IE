# from fake_news_detection import FinancialNewsCredibilityAnalyzer
# from news_sentiment import FinancialNewsAnalyzer
# from price_predictor import ARIMAStockPredictionModel, LSTMStockPricePredictor
#
# import yfinance
# import time
#
#
# # news_article = input("news article about the company (also try containing the source url at the end of the article): ")
# news_article = "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"
# ticker = input("company symbol: ")
#
# # historical_data = yfinance.download(ticker)
# start_time = time.time()
#
# start_sentiment = time.time()
# print("sentiment analyzer")
# sentiment_analyzer = FinancialNewsAnalyzer()
# sentiment_result = sentiment_analyzer.analyze_sentiment(news_article)
# print(f"sentiment: {time.time() - start_sentiment}")
#
# start_fake = time.time()
# print("Fake news analyzer")
# credibility_analyzer = FinancialNewsCredibilityAnalyzer()
# credibility_score = credibility_analyzer.analyze(news_article)['credibility_score']
# print(f"fake: {time.time() - start_fake}")
#
# print(sentiment_result['sentiment_score'], credibility_score)
# #print(type(credibility_score), type(sentiment_result['sentiment_score']))
#
# weighted_credible_sentiment = sentiment_result['sentiment_score'] * credibility_score
# print(f"weighted sentiment {weighted_credible_sentiment}")
#
# prediction_model = ARIMAStockPredictionModel()
# predictions, dates = prediction_model.run_stock_prediction(ticker)
#
# adjusted_predictions, adjusted_predictions_dates = prediction_model.run_stock_prediction_overall_sentiment(weighted_credible_sentiment, ticker)
#
# print("Raw predictions")
# for i in range(len(predictions)):
#     print(f"price {predictions[i]} at date {dates[i]}")
# print(f"TIME ELAPSED RUNNING THE 2 MODELS: {time.time() - start_time} seconds")
#
# print(">>>>>Adjusted predictions<<<<<")
# for i in range(len(adjusted_predictions)):
#     print(f"price {adjusted_predictions[i]} at date {adjusted_predictions_dates[i]}")


from fake_news_detection import FinancialNewsCredibilityAnalyzer
from news_sentiment import FinancialNewsAnalyzer
from price_predictor import ARIMAStockPredictionModel
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import time

def get_sentiment_result(news_text):
    sentiment_analyzer = FinancialNewsAnalyzer()
    print("sentiment analyzer inited")
    return sentiment_analyzer.analyze_sentiment(news_text)

def get_credibility_result(news_text):
    credibility_analyzer = FinancialNewsCredibilityAnalyzer()
    print("credibility analyzer inited")
    return credibility_analyzer.analyze(news_text)['credibility_score']

def fetch_stock_data(ticker, train_start='2010-01-01', train_end='2020-12-31',
                       val_start='2021-01-01', val_end='2023-01-01'):
    start_fetching_time = time.time()
    if ticker is None:
        raise ValueError("Ticker symbol not set. Use set_ticker() method first.")

    data = yf.download(ticker, start=train_start, end=val_end)
    data = data.asfreq('B')
    # data = data.fillna(method='ffill').fillna(method='bfill')
    data = data.ffill().bfill()
    training_data = data['Close'][train_start:train_end]
    validation_data = data['Close'][val_start:val_end]

    print(f"stock data fetched in {time.time() - start_fetching_time}")
    return (ticker, data, training_data, validation_data)

if __name__ == '__main__':
    news_article = "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"
    ticker = input("Company symbol: ")

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        stock_data_future = executor.submit(fetch_stock_data, ticker, train_start='2010-01-01', train_end='2020-12-31', val_start='2021-01-01', val_end='2023-01-01')
        fake_time = time.time()
        credibility_future = executor.submit(get_credibility_result, news_article)
        print(f"credibility model {time.time() - fake_time}")
        sentiment_time = time.time()
        sentiment_future = executor.submit(get_sentiment_result, news_article)
        print(f"sentiment model {time.time() - sentiment_time}")

        sentiment_result = sentiment_future.result()
        credibility_result = credibility_future.result()
        stock_data_result = stock_data_future.result()

    print(sentiment_result['sentiment_score'], credibility_result)

    weighted_credible_sentiment = sentiment_result['sentiment_score'] * credibility_result
    print(f"Weighted sentiment: {weighted_credible_sentiment}")

    # Run the prediction model
    prediction_model = ARIMAStockPredictionModel()
    predictions, dates = prediction_model.run_stock_prediction(stock_data_result)
    #adjusted_predictions, adjusted_predictions_dates = prediction_model.run_stock_prediction_overall_sentiment(weighted_credible_sentiment, ticker)

    print("Raw predictions")
    for i in range(len(predictions)):
        print(f"Price {predictions[i]} at date {dates[i]}")

    print(f"TIME ELAPSED RUNNING THE 2 MODELS: {time.time() - start_time} seconds")

    # print(">>>>> Adjusted predictions <<<<<")
    # for i in range(len(adjusted_predictions)):
    #     print(f"Price {adjusted_predictions[i]} at date {adjusted_predictions_dates[i]}")

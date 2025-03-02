from fake_news_detection import FinancialNewsCredibilityAnalyzer
from news_sentiment import FinancialNewsAnalyzer
from price_predictor import ARIMAStockPredictionModel, LSTMStockPricePredictor

import yfinance


# news_article = input("news article about the company (also try containing the source url at the end of the article): ")
news_article = "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"
ticker = input("company symbol: ")

# historical_data = yfinance.download(ticker)

print("sentiment analyzer")
sentiment_analyzer = FinancialNewsAnalyzer()
sentiment_result = sentiment_analyzer.analyze_sentiment(news_article)

print("Fake news analyzer")
credibility_analyzer = FinancialNewsCredibilityAnalyzer()
credibility_score = credibility_analyzer.analyze(news_article)['credibility_score']

print(sentiment_result['sentiment_score'], credibility_score)
#print(type(credibility_score), type(sentiment_result['sentiment_score']))

weighted_credible_sentiment = sentiment_result['sentiment_score'] * credibility_score
print(f"weighted sentiment {weighted_credible_sentiment}")

prediction_model = ARIMAStockPredictionModel()
predictions, dates = prediction_model.run_stock_prediction(ticker)

adjusted_predictions, adjusted_predictions_dates = prediction_model.run_stock_prediction_overall_sentiment(weighted_credible_sentiment, ticker)

print("Raw predictions")
for i in range(len(predictions)):
    print(f"price {predictions[i]} at date {dates[i]}")

print(">>>>>Adjusted predictions<<<<<")
for i in range(len(adjusted_predictions)):
    print(f"price {adjusted_predictions[i]} at date {adjusted_predictions_dates[i]}")
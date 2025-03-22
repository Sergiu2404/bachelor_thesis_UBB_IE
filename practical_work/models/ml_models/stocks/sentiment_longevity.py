import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(finbert_model_name)
finbert_model = BertForSequenceClassification.from_pretrained(finbert_model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy().flatten()

    sentiment_score = -1 * probabilities[0] + 0 * probabilities[1] + 1 * probabilities[2]
    return sentiment_score

def extract_features(text):
    sentiment = predict_sentiment(text)

    sia = SentimentIntensityAnalyzer()
    sentiment_vader = sia.polarity_scores(text)["compound"]

    text_length = len(text.split())

    word_count = len(text.split())

    return [sentiment, sentiment_vader, text_length, word_count]

data = [
    ("Stock prices are expected to rise with new market changes.", 0.2),
    ("The company reported a significant quarterly loss but plans to recover soon.", 0.4),
    ("A major new economic policy announced by the Federal Reserve is expected to change the market dynamics.", 0.9),
    ("The company just launched a new product, which is a short-term marketing campaign.", 0.3),
    ("Investors are optimistic about the potential merger, which could take years to complete.", 1.0),
]

df = pd.DataFrame(data, columns=["text", "longevity"])

features = df["text"].apply(extract_features).tolist()
X = np.array(features)

y = np.array(df["longevity"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

new_article = "The company is expanding its operations into new markets, with long-term growth expected."

new_article_features = extract_features(new_article)

longevity_prediction = rf_model.predict([new_article_features])
print(f"Predicted Longevity Score: {longevity_prediction[0]}")

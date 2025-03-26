import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy

nltk.download('vader_lexicon')
nltk.download('punkt')


class EnhancedLongevityPredictor:
    def __init__(self):
        self.finbert_model_name = "yiyanghkust/finbert-tone"
        #self.finbert_model_name = "distilbert-base-uncased"
        self.finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(self.finbert_model_name)

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.sia = SentimentIntensityAnalyzer()

    def predict_finbert_sentiment(self, text):
        inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy().flatten()

        sentiment_score = 0 * probabilities[0] + 1 * probabilities[1]
        #sentiment_score = -1 * probabilities[0] + 0 * probabilities[1] + 1 * probabilities[2]
        return sentiment_score

    def extract_advanced_features(self, text):
        finbert_sentiment = self.predict_finbert_sentiment(text)
        vader_sentiment = self.sia.polarity_scores(text)["compound"]
        textblob_sentiment = TextBlob(text).sentiment.polarity

        doc = self.nlp(text)

        entity_count = len(list(doc.ents))
        noun_phrase_count = len([chunk for chunk in doc.noun_chunks])

        avg_word_length = np.mean([len(token.text) for token in doc])
        unique_word_ratio = len(set(token.text.lower() for token in doc)) / len(list(doc))

        temporal_keywords = ['future', 'long-term', 'year', 'decade', 'strategy', 'plan', 'roadmap']
        temporal_keyword_count = sum(1 for word in temporal_keywords if word in text.lower())

        word_count = len(word_tokenize(text))
        complex_word_count = len([token for token in doc if len(token.text) > 6])

        features = [
            finbert_sentiment,
            vader_sentiment,
            textblob_sentiment,
            entity_count,
            noun_phrase_count,
            avg_word_length,
            unique_word_ratio,
            temporal_keyword_count,
            word_count,
            complex_word_count
        ]

        return features

    def train_model(self, df):
        features = df["text"].apply(self.extract_advanced_features).tolist()
        X = np.array(features)
        y = np.array(df["longevity"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        models = [
            ('XGBoost', XGBRegressor(n_estimators=200, learning_rate=0.05)),
            ('RandomForest', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=200, random_state=42))
        ]

        best_model = None
        best_score = float('inf')

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"{name} Results:")
            print(f"Mean Squared Error: {mse}")
            print(f"Mean Absolute Error: {mae}")
            print(f"R2 Score: {r2}\n")

            if mse < best_score:
                best_model = model
                best_score = mse

        return best_model, scaler

    def predict_longevity(self, model, scaler, text):
        features = self.extract_advanced_features(text)
        features_scaled = scaler.transform([features])

        longevity_prediction = model.predict(features_scaled)
        return longevity_prediction[0]


if __name__ == "__main__":
    df = pd.read_csv("../sentiment_datasets/longevity_dataset.csv")
    df.drop_duplicates(inplace=True)

    predictor = EnhancedLongevityPredictor()
    best_model, scaler = predictor.train_model(df)

    # new_article = "The company wants to expand its headquarters to Berlin. This will be a move with impact for the next decades, says the executive chief."
    new_article = "The company wants to keep raising the wages of their employees, says the executive chief."
    longevity_score = predictor.predict_longevity(best_model, scaler, new_article)
    print(f"Predicted Longevity Score: {longevity_score}")

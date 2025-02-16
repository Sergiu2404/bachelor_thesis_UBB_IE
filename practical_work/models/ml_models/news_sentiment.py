# import feedparser
# from transformers import pipeline
#
# class FinBertNewsSentiment:
#     def __init__(self, symbol, keyword):
#         self.symbol = symbol
#         self.keyword = keyword.lower()
#         self.pipe = pipeline("text-classification", model="ProsusAI/finbert")
#         self.rss_url = f'https://finance.yahoo.com/rss/headline?s={self.symbol}'
#         self.total_score = 0
#         self.num_articles = 0
#
#     def analyze_news(self):
#         feed = feedparser.parse(self.rss_url)
#
#         for entry in feed.entries:
#             if self.keyword not in entry.summary.lower():
#                 continue
#
#             print(f"title: {entry.title}")
#             print(f"link: {entry.link}")
#             print(f"published: {entry.published}")
#             print(f"summary: {entry.summary}")
#
#             sentiment = self.pipe(entry.summary)[0]
#             print(f"Sentiment: {sentiment['label']}, score: {sentiment['score']}")
#             print("-" * 20)
#
#             if sentiment['label'] == 'positive':
#                 self.total_score += sentiment['score']
#                 self.num_articles += 1
#             elif sentiment['label'] == 'negative':
#                 self.total_score -= sentiment['score']
#                 self.num_articles += 1
#
#     def get_final_sentiment(self):
#         if self.num_articles == 0:
#             return "No relevant articles found."
#
#         final_score = self.total_score / self.num_articles
#         sentiment_label = 'positive' if final_score >= 0.15 else 'negative' if final_score <= -0.15 else 'neutral'
#         return f"Overall sentiment for news regarding {self.symbol} is {final_score}, so news is mostly {sentiment_label}"


# news_sentiment = FinBertNewsSentiment("META", "meta")
# news_sentiment.analyze_news()
# print(news_sentiment.get_final_sentiment())


import re
import glob
from tqdm import tqdm
import pickle
from pathlib import Path

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextAnalysisPipeline:
    def __init__(self, file_path_pattern=None):
        self.file_path_pattern = file_path_pattern
        self.df = None
        self.sia = SentimentIntensityAnalyzer()
        self.is_trained = False

        # Ensure NLTK resources are available
        required_resources = ['punkt', 'averaged_perceptron_tagger',
                              'maxent_ne_chunker', 'words', 'vader_lexicon']
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading required NLTK resource: {resource}")
                nltk.download(resource)

    def natural_sort(self, files):
        return sorted(files, key=lambda x: [int(text) if text.isdigit() else text.lower()
                                            for text in re.split(r'([0-9]+)', x)])

    def load_data(self):
        print("Loading training data...")
        all_files = glob.glob(self.file_path_pattern)
        sorted_files = self.natural_sort(all_files)
        self.df = pd.concat([pd.read_csv(f) for f in sorted_files], axis=0)
        print(f"Loaded {len(self.df)} reviews for training.")

    def process_text(self, text):
        tokens = word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.chunk.ne_chunk(tagged)
        return tokens, tagged, entities

    def train_model(self):
        """Train the model on the full dataset and save results"""
        if self.file_path_pattern is None:
            raise ValueError("No training data path provided. Please initialize with file_path_pattern.")

        if self.df is None:
            self.load_data()

        print("Analyzing sentiment for all reviews...")
        results = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            text = row["Text"]
            sentiment = self.sia.polarity_scores(text)
            results.append(sentiment)

        self.df['sentiment'] = results
        self.is_trained = True
        print("Training completed!")

        # Calculate and store average sentiments for reference
        self.avg_sentiments = {
            'compound': self.df['sentiment'].apply(lambda x: x['compound']).mean(),
            'pos': self.df['sentiment'].apply(lambda x: x['pos']).mean(),
            'neg': self.df['sentiment'].apply(lambda x: x['neg']).mean(),
            'neu': self.df['sentiment'].apply(lambda x: x['neu']).mean()
        }

        return self.df

    def predict(self, text):
        """Analyze sentiment for a single piece of text"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        sentiment = self.sia.polarity_scores(text)

        # Compare with average sentiments if trained
        if self.is_trained:
            comparison = {
                'sentiment': sentiment,
                'comparison_to_average': {
                    'compound': 'above average' if sentiment['compound'] > self.avg_sentiments[
                        'compound'] else 'below average',
                    'positivity': 'above average' if sentiment['pos'] > self.avg_sentiments['pos'] else 'below average',
                    'negativity': 'above average' if sentiment['neg'] > self.avg_sentiments['neg'] else 'below average'
                }
            }
            return comparison
        return {'sentiment': sentiment}

    def predict_batch(self, texts):
        """Analyze sentiment for multiple texts"""
        return [self.predict(text) for text in texts]


pipeline = TextAnalysisPipeline("./fake_news_datasets/amazon-fine-reviews-dataset-splitted/Reviews*")
pipeline.train_model()

# Example predictions
example_texts = [
    "This product is absolutely amazing! I love everything about it.",
    "The quality was okay, but not worth the price.",
    "Terrible product, complete waste of money."
]

print("\nAnalyzing example texts:")
for text in example_texts:
    result = pipeline.predict(text)
    print(f"\nText: {text}")
    print(f"Sentiment scores: {result['sentiment']}")
    print(f"Comparison to average: {result['comparison_to_average']}")







# nltk.download('punkt')
# nltk.download('stopwords')


class VaderSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.financial_lexicon = {
            "bullish": 2.5,
            "bearish": -2.5,
            "upgrade": 1.5,
            "downgrade": -1.5,
            "profit": 2.0,
            "loss": -2.0,
            "rally": 1.8,
            "plunge": -2.2
        }
        self.stop_words = set(stopwords.words('english'))
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return " ".join(tokens)

    def run_model(self, texts):
        preprocessed_texts = []

        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            preprocessed_texts.append(preprocessed_text)

        self.sia.lexicon.update(self.financial_lexicon)

        for text in preprocessed_texts:
            print(text, "->", self.sia.polarity_scores(text))


example_texts = [
    "The stock market experienced a bullish rally today.",
    "Company reports a significant loss in Q4 earnings.",
    "Analyst upgrades the stock to a strong buy."
]

# vader_sentiment_analyszer = VaderSentimentAnalyzer()
# vader_sentiment_analyszer.run_model(example_texts)
import os
import random

import pandas as pd
import string
import numpy as np
import re

import pickle
import requests
from bs4 import BeautifulSoup
import warnings

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')


class FinancialNewsAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("download nltk resources...")
            # nltk.download('punkt')
            # nltk.download('stopwords')
            # nltk.download('wordnet')

        self.dataset_path = "fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0"

        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {"not", "never", "no", "without", "hardly", "barely"}

        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        self.model = None

        self.financial_dictionary_classifier = self.load_financial_dictionary_classifier()

    def get_wordnet_pos(self, word):
        """Convert POS tag to format used by WordNetLemmatizer"""
        tag = pos_tag([word])[0][1][0].upper()  # get first letter of POS tag
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # def lexicon_score(self, text):
    #     """Calculate sentiment score using financial lexicon, considering negations and analyst rating changes."""
    #     text = text.lower()
    #     tokens = word_tokenize(text)
    #     score = 0
    #     negate = False
    #
    #     # define phrase based sentiment adjustments
    #     phrase_sentiments = {
    #         "from buy to hold": -2.0, "from hold to sell": -2.5, "from buy to sell": -3.0,
    #         "downgraded to hold": -2.0, "downgraded to sell": -3.0, "cut to hold": -2.0, "cut to sell": -3.0,
    #
    #         "from sell to hold": 2.0, "from hold to buy": 2.5, "from sell to buy": 3.0,
    #         "upgraded to buy": 3.0, "upgraded to hold": 2.0, "raised to buy": 3.0
    #     }
    #
    #     # check for phrases
    #     for phrase, sentiment in phrase_sentiments.items():
    #         if phrase in text:
    #             score += sentiment
    #
    #     for token in tokens:
    #         lemmatized_token = self.lemmatizer.lemmatize(token)
    #
    #         if lemmatized_token in self.negation_words:
    #             negate = True  # found negation
    #             continue
    #
    #         if lemmatized_token in self.financial_dictionary_classifier:
    #             sentiment = self.financial_dictionary_classifier[lemmatized_token]
    #             score += -sentiment if negate else sentiment  # flip the sentiment if negation applies
    #             negate = False  # reset negation after applying
    #
    #     return score / max(len(tokens), 1)

    def lexicon_score(self, text):
        """Calculate sentiment score using financial lexicon, considering negations and analyst rating changes."""
        text = text.lower()
        tokens = word_tokenize(text)
        score = 0
        negate = False
        negation_scope = []
        stop_words = {".", ",", ";", ":", "!", "?"}  # they end negation

        # Define phrase-based sentiment adjustments
        phrase_sentiments = {
            "from buy to hold": -3.0, "from hold to sell": -3.5, "from buy to sell": -3.5,
            "downgraded to hold": -3.0, "downgraded to sell": -3.5, "cut to hold": -3.5, "cut to sell": -3.5,
            "from sell to hold": 2.0, "from hold to buy": 2.2, "from sell to buy": 2.4,
            "upgraded to buy": 2.0, "upgraded to hold": 2.2, "raised to buy": 2.4
        }

        # Check for phrases first
        for phrase, sentiment in phrase_sentiments.items():
            if phrase in text:
                score += sentiment

        for token in tokens:
            lemmatized_token = self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token))

            if lemmatized_token in self.negation_words:
                negate = True  # Activate negation
                negation_scope = []  # Reset scope
                continue

            if negate:
                negation_scope.append(lemmatized_token)

            if lemmatized_token in self.financial_dictionary_classifier:
                sentiment = self.financial_dictionary_classifier[lemmatized_token]

                if negate:
                    # Flip the entire negation scope
                    for neg_word in negation_scope:
                        if neg_word in self.financial_dictionary_classifier:
                            sentiment = -self.financial_dictionary_classifier[neg_word]
                            score += sentiment
                    negation_scope = []  # Clear scope after applying negation
                    negate = False  # Reset negation

                else:
                    score += sentiment

            # Reset negation at punctuation
            if token in stop_words:
                negate = False
                negation_scope = []

        return score / max(len(tokens), 1)

    def load_financial_dictionary_classifier(self):
        """Load or create a financial-specific sentiment lexicon"""
        financial_positive = [
            'beat', 'boost', 'exceed', 'surprisingly', 'grow', 'up', 'rise', 'gain', 'profitable',
            'earn', 'strong', 'strength', 'higher', 'high', 'rally', 'bullish', 'outperform',
            'opportunity', 'success', 'improve', 'breakthrough', 'progress', 'upgrade', 'increase', 'win', 'reward',
            'advance', 'progress', 'soar', 'climb', 'ascent', 'great', 'amazing'
        ]

        financial_negative = [
            'miss', 'disappoint', 'decline', 'decrease', 'loss', 'negative', 'descent', 'low',
            'weak', 'drop', 'fall', 'bearish', 'underperform', 'risk', 'dive', 'problem'
            'warning', 'fail', 'bankruptcy', 'investigation', 'lawsuit', 'litigation', 'pressure'
            'concern', 'caution', 'downgrade', 'decrease', 'lose', 'challenge', 'bearish', 'degeneration'
        ]

        financial_neutral = [
            'unchanged', 'neutral', 'maintain', 'expect', 'keep'
            'estimate', 'guidance', 'target', 'announce', 'report', 'quarter',
            'fiscal', 'year', 'slightly', 'slight', 'pretty', 'slow', 'prudent', 'insignificant', 'unsignificant'
        ]

        lexicon = {}
        for word in financial_positive:
            lexicon[self.lemmatizer.lemmatize(word)] = random.uniform(1, 1.5)
        for word in financial_negative:
            lexicon[self.lemmatizer.lemmatize(word)] = random.uniform(-3, -3.5)
        for word in financial_neutral:
            lexicon[self.lemmatizer.lemmatize(word)] = 0

        return lexicon

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = text.lower()
        # remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # remove html
        text = re.sub(r'<.*?>', '', text)
        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # tokenize
        tokens = word_tokenize(text)

        # lemmatize tokens and remove stopwords
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(lemmatized_tokens)

        # remove stopwords and lemmatize
        # processed_tokens = [
        #     self.lemmatizer.lemmatize(token) for token in tokens
        #     if token not in self.stop_words and len(token) > 2
        # ]
        #
        # return ' '.join(processed_tokens)

    def load_financial_phrasebank(self):
        """Load and preprocess the Financial PhraseBank dataset."""
        data = []
        files = [file for file in os.listdir(self.dataset_path) if file.startswith("Sentences")]

        print("Loading Financial PhraseBank dataset...")

        # for file_name in files:
        #     file_path = os.path.join(self.dataset_path, file_name)
        #     print(f"processing file {file_name}...")
        #
        #     with open(file_path, "r", encoding="ISO-8859-1") as file:
        #         for line in file:
        #             try:
        #                 text, sentiment = line.rsplit("@", 1)
        #                 sentiment = sentiment.strip()
        #                 label = {"neutral": 0, "positive": 1, "negative": -1}.get(sentiment, 0)
        #                 data.append({"text": text.strip(), "sentiment": label})
        #             except ValueError:
        #                 print(f"Skipping malformed line in {file_name}: {line}")
        #
        # dataset = pd.DataFrame(data)
        # print(f"Total samples loaded: {len(dataset)}")
        # return dataset


        data = []
        with open("fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r", encoding="ISO-8859-1") as file:
            for line in file:
                text, sentiment = line.rsplit("@", 1)
                sentiment = sentiment.strip()
                label = {"neutral": 0, "positive": 1, "negative": -1}.get(sentiment, 0)
                data.append({"text": text.strip(), "sentiment": label})

        dataset = pd.DataFrame(data)
        print(dataset['sentiment'].value_counts())
        return dataset

    # def get_financial_dataset(self):
    #     """Download or create financial news sentiment dataset"""
    #     # Check if we have a saved dataset
    #     if os.path.exists('fake_news_datasets/financial_news_sentiment.csv'):
    #         print("Loading existing dataset...")
    #         return pd.read_csv('fake_news_datasets/financial_news_sentiment.csv')
    #
    #     print("Creating new financial news sentiment dataset...")
    #
    #     # Financial PhraseBank dataset
    #     try:
    #         url = "https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip"
    #         # For this example, we'll assume we have it downloaded
    #         # In practice, you would download and extract the zip file
    #
    #         # Placeholder data similar to FinancialPhraseBank format
    #         data = [
    #             {'text': 'The company reported strong earnings this quarter, beating analyst expectations.',
    #              'sentiment': 1},
    #             {'text': 'The stock plummeted after the company missed revenue targets.', 'sentiment': -1},
    #             {'text': 'The company maintained its previous guidance for the fiscal year.', 'sentiment': 0},
    #             {'text': 'Profits soared by 25% year-over-year, driving the stock to new highs.', 'sentiment': 1},
    #             {'text': 'The CEO announced layoffs affecting 15% of the workforce.', 'sentiment': -1}
    #         ]
    #         dataset1 = pd.DataFrame(data)
    #     except:
    #         # Fallback sample data
    #         data = [
    #             {'text': 'The company reported strong earnings this quarter, beating analyst expectations.',
    #              'sentiment': 1},
    #             {'text': 'The stock plummeted after the company missed revenue targets.', 'sentiment': -1},
    #             {'text': 'The company maintained its previous guidance for the fiscal year.', 'sentiment': 0}
    #         ]
    #         dataset1 = pd.DataFrame(data)
    #
    #     # create synthetic data using our financial lexicon
    #     synthetic_data = []
    #     # Positive examples
    #     for _ in range(300):
    #         positive_words = np.random.choice(
    #             [word for word, score in self.financial_dictionary_classifier.items() if score > 0],
    #             size=np.random.randint(3, 8)
    #         )
    #         text = f"The company {np.random.choice(['reported', 'announced', 'showed'])} " + \
    #                f"{' and '.join(positive_words)} " + \
    #                f"for {np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {np.random.randint(2020, 2024)}."
    #         synthetic_data.append({'text': text, 'sentiment': 1})
    #
    #     # negative examples
    #     for _ in range(300):
    #         negative_words = np.random.choice(
    #             [word for word, score in self.financial_dictionary_classifier.items() if score < 0],
    #             size=np.random.randint(3, 8)
    #         )
    #         text = f"The company {np.random.choice(['reported', 'announced', 'showed'])} " + \
    #                f"{' and '.join(negative_words)} " + \
    #                f"for {np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {np.random.randint(2020, 2024)}."
    #         synthetic_data.append({'text': text, 'sentiment': -1})
    #
    #     # neutral examples
    #     for _ in range(300):
    #         neutral_words = np.random.choice(
    #             [word for word, score in self.financial_dictionary_classifier.items() if score == 0],
    #             size=np.random.randint(3, 8)
    #         )
    #         text = f"The company {np.random.choice(['reported', 'announced', 'showed'])} " + \
    #                f"{' and '.join(neutral_words)} " + \
    #                f"for {np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {np.random.randint(2020, 2024)}."
    #         synthetic_data.append({'text': text, 'sentiment': 0})
    #
    #     dataset2 = pd.DataFrame(synthetic_data)
    #
    #     # combine datasets
    #     combined_dataset = pd.concat([dataset1, dataset2], ignore_index=True)
    #
    #     # add ticker column (random for demonstration)
    #     tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
    #     combined_dataset['ticker'] = np.random.choice(tickers, size=len(combined_dataset))
    #
    #     # preprocess texts
    #     combined_dataset['processed_text'] = combined_dataset['text'].apply(self.preprocess_text)
    #
    #     # save dataset
    #     combined_dataset.to_csv('financial_news_sentiment.csv', index=False)
    #
    #     return combined_dataset

    def train_model(self):
        """Train the sentiment analysis model."""
        dataset = self.load_financial_phrasebank()
        dataset["processed_text"] = dataset["text"].apply(self.preprocess_text)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["processed_text"], dataset["sentiment"], test_size=0.2, random_state=42
        )

        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # train model
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X_train_vec, y_train)

        # evaluate performance
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # save model
        with open("financial_sentiment_model.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

        return accuracy

    def load_model(self):
        """Load a pre-trained model"""
        try:
            with open('financial_sentiment_model.pkl', 'rb') as f:
                self.vectorizer, self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            print("No pre-trained model found. Training a new model...")
            self.train_model()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def fetch_news_for_ticker(self, ticker):
        """Fetch recent news articles for a ticker symbol
        This is a placeholder. In production, you'd use a proper API.
        """
        # Placeholder - in production use a real news API
        print(f"Fetching news for {ticker}...")

        # Sample news articles
        sample_news = [
            {
                'title': f"{ticker} Reports Quarterly Earnings Above Expectations",
                'content': f"{ticker} reported quarterly earnings that exceeded analyst expectations, with revenue growing 15% year-over-year. The company also raised its full-year guidance."
            },
            {
                'title': f"Analyst Downgrades {ticker} Citing Competitive Pressures",
                'content': f"An analyst at a major investment bank downgraded {ticker} from Buy to Hold, citing increasing competitive pressures and margin concerns in the coming quarters."
            },
            {
                'title': f"{ticker} Announces New Product Launch",
                'content': f"{ticker} unveiled its newest product line at an industry conference. Executives expect the new offerings to be revenue-neutral in the short term."
            },
            {
                'title': f"{ticker} Announces New Record Sales",
                'content': f"{ticker} announced yesterday high record sales for the last trimester. Investors expect new records to be hit by the company in the future."
            },
            {
                'title': f"{ticker} Some Negative Records In Profit",
                'content': f"{ticker} just announced that for the last semester they got the lowest profit in the last 10 years, which is a negative record for the company."
            },
            {
                'title': f"{ticker} Gives New Declarations",
                'content': f"{ticker} just announced they do not think they got the lowest profit in the last 10 years, which is a negative record for the company."
            }
        ]

        return sample_news

    def analyze_sentiment(self, text, ticker=None):
        """Analyze sentiment of a financial news article"""
        # Ensure model is loaded
        if self.model is None or self.vectorizer is None:
            self.load_model()

        processed_text = self.preprocess_text(text)

        # get prediction
        X_vec = self.vectorizer.transform([processed_text])
        model_score = self.model.predict_proba(X_vec)[0]

        # Convert 3-class probabilities to a single score between -1 and 1
        # Assuming classes are ordered as [-1, 0, 1] or [0, 1, 2]
        if len(model_score) == 3:  # 3-class model
            # Convert [neg_prob, neutral_prob, pos_prob] to a single score
            score = model_score[2] - model_score[0]  # pos_prob - neg_prob
        else:  # 2-class model
            score = model_score[1] * 2 - 1  # Convert 0-1 to -1 to 1

        lex_score = self.lexicon_score(processed_text)
        combined_score = 0.7 * score + 0.3 * lex_score

        if combined_score > 0.25:
            sentiment = "Positive"
        elif combined_score < -0.25:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        result = {
            'ticker': ticker,
            'raw_text': text,
            'processed_text': processed_text,
            'sentiment_score': round(combined_score, 3),
            'sentiment_class': sentiment,
            'model_confidence': max(model_score) if len(model_score) == 3 else max(model_score[0], model_score[1])
        }

        return result

def analyze_ticker_news(ticker, custom_article=None):
    analyzer = FinancialNewsAnalyzer()

    # check if model trained/loaded
    if not os.path.exists('financial_sentiment_model.pkl'):
        print("Training new sentiment model...")
        analyzer.train_model()
    else:
        analyzer.load_model()

    if custom_article:
        result = analyzer.analyze_sentiment(custom_article, ticker)
        print(f"\nAnalysis for custom article about {ticker}:")
        print(f"Sentiment: {result['sentiment_class']} (Score: {result['sentiment_score']})")
        print(f"Confidence: {result['model_confidence']:.2f}")
        return result

    articles = analyzer.fetch_news_for_ticker(ticker)
    results = []

    print(f"\nSentiment Analysis for {ticker} news:")
    for i, article in enumerate(articles):
        result = analyzer.analyze_sentiment(
            f"{article['title']}. {article['content']}",
            ticker
        )
        results.append(result)

        print(f"\nArticle {i + 1}:")
        print(f"Headline: {article['title']}")
        print(f"Sentiment: {result['sentiment_class']} (Score: {result['sentiment_score']})")

    # Calculate average sentiment
    avg_score = sum(r['sentiment_score'] for r in results) / len(results)
    print(f"\nOverall {ticker} sentiment: {avg_score:.3f}")

    return results

if __name__ == "__main__":
    analyze_ticker_news("AAPL")

    custom_news = """
    NVDA shares jumped 8% after the company reported blockbuster earnings, 
    beating Wall Street expectations by a wide margin. Revenue from AI chips 
    tripled year-over-year, and the CEO announced plans to increase production 
    capacity to meet surging demand.
    """
    analyze_ticker_news("NVDA", custom_news)





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
                #nltk.download(resource)

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


# pipeline = TextAnalysisPipeline("./fake_news_datasets/amazon-fine-reviews-dataset-splitted/Reviews*")
# pipeline.train_model()
#
# # Example predictions
# example_texts = [
#     "This product is absolutely amazing! I love everything about it.",
#     "The quality was okay, but not worth the price.",
#     "Terrible product, complete waste of money."
# ]
#
# print("\nAnalyzing example texts:")
# for text in example_texts:
#     result = pipeline.predict(text)
#     print(f"\nText: {text}")
#     print(f"Sentiment scores: {result['sentiment']}")
#     print(f"Comparison to average: {result['comparison_to_average']}")







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
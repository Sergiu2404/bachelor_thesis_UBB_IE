import os
import random
import warnings
import re

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



class FinancialNewsAnalyzer:
    def __init__(self):
        try:
            print()
            # nltk.data.find('tokenizers/punkt')
            # nltk.data.find('corpora/stopwords')
            # nltk.data.find('corpora/wordnet')
        except LookupError:
            print("download nltk resources...")
            # nltk.download('punkt')
            # nltk.download('stopwords')
            # nltk.download('wordnet')

        self.dataset_path = "fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0"

        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {
            "n't", "not", "never", "no", "without", "hardly", "barely",
            "fail", "unable", "unlikely", "absence", "prevent", "lack",
            "deny", "dismiss", "reject", "cannot"
        }

        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        self.model = None

        self.financial_dictionary_classifier = self.load_financial_dictionary_classifier()

    def get_wordnet_pos(self, word):
        """Convert POS tag to format used by WordNetLemmatizer"""
        tag = pos_tag([word])[0][1][0].upper()  # get first letter of POS tag
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def load_financial_dictionary_classifier(self):
        """Load or create a financial-specific sentiment lexicon"""
        financial_positive = [
            'beat', 'boost', 'exceed', 'surprisingly', 'grow', 'up', 'rise', 'gain', 'profitable', 'hike', 'well',
            'earn', 'strong', 'strength', 'higher', 'high', 'rally', 'bullish', 'outperform', 'surpass', 'good',
            'opportunity', 'success', 'improve', 'breakthrough', 'progress', 'upgrade', 'increase', 'win', 'reward',
            'advance', 'progress', 'soar', 'climb', 'ascent', 'great', 'amazing', 'above', 'expansion', 'expand',
            'optimistic', 'wide', 'jump', 'double', 'triple', 'twice', 'upside'
        ]

        financial_negative = [
            'miss', 'disappoint', 'decline', 'decrease', 'loss', 'negative', 'descent', 'low', 'slow', 'down', 'slowdown',
            'weak', 'drop', 'fall', 'bearish', 'underperform', 'risk', 'dive', 'problem', 'below', 'downside', 'headwind',
            'warning', 'fail', 'bankruptcy', 'investigation', 'lawsuit', 'litigation', 'pressure', 'fear', 'recession', 'bad',
            'concern', 'caution', 'downgrade', 'decrease', 'lose', 'challenge', 'bearish', 'degeneration', 'half', 'downside'
        ]

        financial_neutral = [
            'unchanged', 'neutral', 'maintain', 'expect', 'keep'
            'estimate', 'guidance', 'target', 'announce', 'report', 'quarter',
            'fiscal', 'year', 'slightly', 'slight', 'pretty', 'slow', 'prudent', 'insignificant', 'unsignificant'
        ]

        lexicon = {}
        for word in financial_positive:
            lexicon[self.lemmatizer.lemmatize(word)] = random.uniform(1.5, 2)
        for word in financial_negative:
            lexicon[self.lemmatizer.lemmatize(word)] = random.uniform(-2, -1.5)
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
        #text = text.translate(str.maketrans('', '', string.punctuation))
        # tokenize
        tokens = word_tokenize(text)

        # lemmatize tokens and remove stopwords
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens if len(token) > 1]
        return ' '.join(lemmatized_tokens)

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

    def train_model(self):
        """Train the sentiment analysis model."""
        dataset = self.load_financial_phrasebank()
        dataset["processed_text"] = dataset["text"].apply(self.preprocess_text)

        X_train, X_test, y_train, y_test = train_test_split(
            dataset["processed_text"], dataset["sentiment"], test_size=0.2, random_state=42
        )

        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

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
        print(f"Fetching news for {ticker}...")

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
                'content': f"{ticker} just announced they don't think they got the lowest profit in the last 10 years."
            }
        ]

        return sample_news

    def detect_negation_scope(self, text):
        """
        Identifies words affected by negation in financial text.
        Returns a set of words that should have their sentiment flipped.
        """
        tokens = word_tokenize(text.lower())
        negated_words = set()

        # typical negation scope terminators
        scope_terminators = {"but", "however", "nevertheless", "yet", "although", "though",
                             ".", ",", ";", ":", "!", "?"}

        negation_active = False
        negation_distance = 0
        max_negation_distance = 10  # max range of words affected by negation

        for i, token in enumerate(tokens):
            # check if token is negation word
            if token in self.negation_words:
                negation_active = True
                negation_distance = 0
                continue

            # if negation active, mark words for sentiment flipping
            if negation_active:
                negation_distance += 1

                # check if it should end the negation scope
                if token in scope_terminators or negation_distance > max_negation_distance:
                    negation_active = False
                else:
                    #add meaningful words to negated set (not stopwords)
                    if token not in self.stop_words:
                        #negated_words.add(token)
                        negated_words.add(self.lemmatizer.lemmatize(token))

        return negated_words

    # def lexicon_score(self, text):
    #     """
    #     Calculate sentiment score with improved downgrade/upgrade detection and negation handling.
    #     """
    #     text = text.lower()
    #     tokens = word_tokenize(text)
    #     score = 0
    #
    #     phrase_sentiments = {
    #         "from buy to hold": -2.0, "from hold to sell": -2.5, "from buy to sell": -3.0,
    #         "downgrade to hold": -2.0, "downgrade to sell": -3.0, "cut to hold": -2.0, "cut to sell": -3.0,
    #         "downgrade": -1.8,
    #         "from sell to hold": 2.0, "from hold to buy": 2.5, "from sell to buy": 3.0, "reinstates dividend": 1.6, "profit boost": 1.7,
    #         "market leader": 1.8, "top performer": 1.8,
    #         "upgrade to buy": 3.0, "upgrade to hold": 2.0, "raise to buy": 3.0, "acquisition deal": 2.0, "strategic partnership": 2.0,
    #         "upgrade": 1.8, "new product": 1.5, "new offer": 1.5, "new offering": 1.5, "expansion plan": 1.5, "new contract": 1.5,
    #         "n't meet expectation": -2.5, "not profitable": -2.0, "no growth": -1.5,
    #         "n't achieve target": -2.0, "never recover": -1.5,
    #         "competitive pressure": -1.5, "margin concern": -1.7
    #     }
    #
    #     # check for phrases for custom sentiment
    #     for phrase, sentiment in phrase_sentiments.items():
    #         if phrase in text:
    #             score += sentiment
    #
    #     # get words after negation
    #     negated_words = self.detect_negation_scope(text)
    #
    #
    #     for token in tokens:
    #         lemmatized_token = self.lemmatizer.lemmatize(token)
    #
    #         if lemmatized_token in self.financial_dictionary_classifier:
    #             sentiment = self.financial_dictionary_classifier[lemmatized_token]
    #
    #             # flip sentiment if word is in negation scope
    #             if token in negated_words or lemmatized_token in negated_words:
    #                 sentiment = -sentiment
    #
    #             score += sentiment
    #
    #     # higher weights for titles containing downgrades/upgrades
    #     if any(term in text[:50].lower() for term in ["downgrade", "upgrade", "cut", "raise", "record"]):
    #         score *= 1.5  # Amplify sentiment for headlines with rating changes
    #
    #     # normalize score by text length
    #     return score / max(len(tokens), 1)
    def lexicon_score(self, text):
        """
        Calculate sentiment score without normalization but with capping.
        """
        text = text.lower()
        tokens = word_tokenize(text)
        score = 0

        phrase_sentiments = {
                "from buy to hold": -2, "from hold to sell": -2, "from buy to sell": -2,
                "downgrade to hold": -2, "downgrade to sell": -2, "cut to hold": -2, "cut to sell": -2,
                "downgrade": -2,
                "from sell to hold": 2, "from hold to buy": 2, "from sell to buy": 2, "reinstates dividend": 1.5, "profit boost": 1.5,
                "market leader": 2, "top performer": 2,
                "upgrade to buy": 2, "upgrade to hold": 2, "raise to buy": 2, "acquisition deal": 2, "strategic partnership": 2,
                "upgrade": 2, "new product": 1.5, "new offer": 1.5, "new offering": 1.5, "expansion plan": 1.5, "new contract": 1.5,
                "n't meet expectation": -2, "not profitable": -2, "no growth": -1.5,
                "n't achieve target": -2, "never recover": -1.5,
                "competitive pressure": -1.5, "margin concern": -1.5
        }

        # Check for phrases with custom sentiment
        for phrase, sentiment in phrase_sentiments.items():
            if phrase in text:
                score += sentiment

        # Get words after negation
        negated_words = self.detect_negation_scope(text)

        for token in tokens:
            lemmatized_token = self.lemmatizer.lemmatize(token)

            if lemmatized_token in self.financial_dictionary_classifier:
                sentiment = self.financial_dictionary_classifier[lemmatized_token]

                # Flip sentiment if word is in negation scope
                if token in negated_words or lemmatized_token in negated_words:
                    sentiment = -sentiment * 0.3
                    # since these words are negated, it doesn't mean we have to completely flip the sentiment, but also to give a neutrality impact

                score += sentiment

        # Higher weights for titles containing downgrades/upgrades
        if any(term in text[:50].lower() for term in ["downgrade", "upgrade", "cut", "raise", "record"]):
            score *= 1.5  # Amplify sentiment for headlines with rating changes

        # cap the score to prevent extremely high values in long texts
        return max(min(score, 1), -1)


    def analyze_sentiment(self, text):
        """Analyze sentiment with improved handling of downgrades and analyst ratings"""
        if self.model is None or self.vectorizer is None:
            self.load_model()

        title_content = text.split('. ', 1)
        title = title_content[0] if len(title_content) > 1 else ""

        # get processed text but preserve original for negation detection
        processed_text = self.preprocess_text(text)

        # get prediction
        X_vec = self.vectorizer.transform([processed_text])
        model_score = self.model.predict_proba(X_vec)[0]

        # convert probability to score
        if len(model_score) == 3:
            ml_score = model_score[2] - model_score[0]  # pos_prob - neg_prob
        else:
            ml_score = model_score[1] * 2 - 1  # Convert 0-1 to -1 to 1

        negation_present = any(term in word_tokenize(processed_text) for term in self.negation_words)
        # for term in processed_text:
        #     if term in self.negation_words:
        #         print(term)
        if negation_present:
            ml_score *= 0.75

        lex_score = self.lexicon_score(processed_text)

        # increase weight for some special words
        if negation_present:
            combined_score = 0.3 * ml_score + 0.7 * lex_score
        elif any(term in text.lower() for term in ["downgrade", "upgrade", "cut", "raise", "record"]):
            combined_score = 0.5 * ml_score + 0.5 * lex_score
        else:
            combined_score = 0.45 * ml_score + 0.55 * lex_score

        if combined_score > 0.25:
            sentiment = "Positive"
        elif combined_score < -0.25:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        result = {
            'raw_text': text,
            'processed_text': processed_text,
            'ml_score': round(ml_score, 3),
            'lexicon_score': round(lex_score, 3),
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
        result = analyzer.analyze_sentiment(custom_article)
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
    #analyze_ticker_news("AAPL")

    # custom_news = """
    # NVDA shares jumped 8% after the company reported blockbuster earnings,
    # beating Wall Street expectations by a wide margin. Revenue from AI chips
    # tripled year-over-year, and the CEO announced plans to increase production
    # capacity to meet surging demand.
    # """
    # analyze_ticker_news("NVDA", custom_news)
    print(analyze_ticker_news("AAPL", "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"))









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
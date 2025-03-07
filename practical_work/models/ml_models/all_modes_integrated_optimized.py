import os
import random
import re
import warnings
import zipfile
from datetime import timedelta
from urllib.parse import urlparse
import time
import threading
import queue

import joblib
import language_tool_python
import numpy as np
import pandas as pd
import validators
from nltk import word_tokenize
from pmdarima import auto_arima
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

# Global variables for model storage
credibility_model = None
credibility_vectorization = None
sentiment_vectorizer = None
sentiment_model = None
arima_model = None
stock_data = None
ticker = None

# Global lock for printing to prevent output interleaving
print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# =======NEWS CREDIBILITY MODEL=======
language_tool = language_tool_python.LanguageTool('en-US')
SUSPICIOUS_DOMAINS = ['.com.co', '.co.com', '.lo', 'finance-news24',
                      'breaking-finance', 'stockalert', 'investing-secrets',
                      'financial-trends', 'money-news-now', 'wallst-alerts']
CREDIBLE_FINANCIAL_SOURCES = {
    'wsj.com': 0.9, 'bloomberg.com': 0.9, 'ft.com': 0.9,
    'reuters.com': 0.9, 'cnbc.com': 0.85, 'economist.com': 0.85,
    'marketwatch.com': 0.85, 'barrons.com': 0.85, 'forbes.com': 0.85,
    'morningstar.com': 0.85, 'investors.com': 0.85, 'businessinsider.com': 0.8,
    'fool.com': 0.7, 'seekingalpha.com': 0.7, 'yahoo.com/finance': 0.8,
    'cnn.com': 0.85, 'nytimes.com': 0.85
}


def grammatical_score(text):
    matches = language_tool.check(text)
    if len(matches) == 0:
        return 1

    for match in matches:
        thread_safe_print(match)

    return round(len(language_tool.check(text)) / max(len(word_tokenize(text)), 1), 3)


def text_quality_score(text):
    score = 1
    all_caps_patterns = re.findall(r'\b[A-Z]{3,}\b', text)

    all_caps_ratio = len(all_caps_patterns) / max(1, len(text.split()))
    excessive_punctuation_ratio = (text.count('!') + text.count('?') + text.count("  ")) / max(1, len(text))
    grammar_errors_ratio = grammatical_score(text)

    thread_safe_print(all_caps_ratio, excessive_punctuation_ratio, grammar_errors_ratio)
    if all_caps_ratio >= 0.2:
        score *= 0.5
    elif all_caps_ratio >= 0.1:
        score *= 0.7

    if excessive_punctuation_ratio >= 0.03:
        score *= 0.7
    elif excessive_punctuation_ratio >= 0.01:
        score *= 0.9

    score = score * grammar_errors_ratio

    thread_safe_print(score)
    return max(0.1, score)


def domain_legitimacy(url):
    score = 1
    if not url or not validators.url(url):
        score *= 0.4

    parsed_url = urlparse(url)
    protocol = parsed_url.scheme
    domain = parsed_url.netloc

    if protocol == "http":
        score *= 0.7  # reduce the score in case secure http is not used
    for suspicious_domain in SUSPICIOUS_DOMAINS:
        if suspicious_domain in domain or domain in SUSPICIOUS_DOMAINS or len(domain) < 5:
            score *= 0.3  # reduce if domain is suspicious

    return score


def get_credibility(credibility_score):
    credibility_texts = ['Highly Suspicious', 'Unreliable', 'Uncertain', 'Likely Credible', 'Highly Credible']
    credibility_index = int(credibility_score * 5)
    if credibility_index >= 5:
        credibility_index = 4

    return {
        'credibility_score': credibility_score,
        'credibility': credibility_texts[credibility_index]
    }


def extract_urls_from_text(text):
    """Extract all URLs from text content"""
    if not isinstance(text, str):
        return []
    url_pattern = re.compile(r'https?://\S+|www\.\S+|[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\.[a-zA-Z]{2,}')
    return url_pattern.findall(text)


def check_source_credibility(url):
    if not url:
        return 0.8
    domain = urlparse(url).netloc.replace('www.', '')

    if domain in CREDIBLE_FINANCIAL_SOURCES:
        return CREDIBLE_FINANCIAL_SOURCES[domain]
    if any(suspicious_domain in domain for suspicious_domain in SUSPICIOUS_DOMAINS):
        return 0.2
    return 0.5


def preprocess_credibility_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    return text.strip()


def train_credibility_model():
    global credibility_model, credibility_vectorization
    try:
        true = pd.read_csv("./fake_news_datasets/fake_news_datasets/True.csv")
        fake = pd.read_csv("./fake_news_datasets/fake_news_datasets/Fake.csv")
    except Exception:
        true, fake = pd.DataFrame(columns=['text']), pd.DataFrame(columns=['text'])

    true['label'], fake['label'] = 1, 0
    news = pd.concat([true, fake], axis=0).drop(['title', 'subject', 'date'], axis=1, errors='ignore')
    news = news.sample(frac=1).reset_index(drop=True)  # shuffle entire dataset, 100% of the dataset
    news['text'] = news['text'].apply(preprocess_credibility_text)

    x_train, x_test, y_train, y_test = train_test_split(news['text'], news['label'], test_size=0.2, random_state=42)
    credibility_vectorization = TfidfVectorizer()
    credibility_vectorization.fit(x_train)  # convert to numerical vals
    xv_train, xv_test = credibility_vectorization.transform(x_train), credibility_vectorization.transform(
        x_test)  # convert to a sparse matrix using vocab

    credibility_model = LogisticRegression(max_iter=1000)
    credibility_model.fit(xv_train, y_train)
    pred = credibility_model.predict_proba(xv_test)[:, 1]  # output prob for each class

    thread_safe_print(classification_report(y_test, (pred > 0.5).astype(int), zero_division=0))


def save_credibility_model():
    model_path = "E:\\saved_models\\fake_news_detection_model\\credibility_model.zip"
    os.makedirs("E:\\saved_models", exist_ok=True)

    joblib.dump(credibility_model, "E:\\saved_models\\fake_news_detection_model\\credibility_model.pkl")
    joblib.dump(credibility_vectorization, "E:\\saved_models\\fake_news_detection_model\\vectorizer.pkl")

    with zipfile.ZipFile(model_path, 'w') as zip_ref:
        zip_ref.write("E:\\saved_models\\fake_news_detection_model\\credibility_model.pkl", "credibility_model.pkl")
        zip_ref.write("E:\\saved_models\\fake_news_detection_model\\vectorizer.pkl", "vectorizer.pkl")


def load_credibility_model():
    global credibility_model, credibility_vectorization
    thread_safe_print("Loading credibility model...")
    start_time = time.time()

    model_path = "E:\\saved_models\\fake_news_detection_model\\credibility_model.zip"
    if os.path.exists(model_path):
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall("E:\\saved_models")

        credibility_model = joblib.load("E:\\saved_models\\fake_news_detection_model\\credibility_model.pkl")
        credibility_vectorization = joblib.load("E:\\saved_models\\fake_news_detection_model\\vectorizer.pkl")
    else:
        train_credibility_model()
        save_credibility_model()

    elapsed_time = time.time() - start_time
    thread_safe_print(f"Credibility model loaded in {elapsed_time:.2f} seconds")




# =======NEWS SENTIMENT MODEL=======
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

import pickle
from nltk.corpus import stopwords

dataset_path = "fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0"
stop_words = set(stopwords.words('english'))
negation_words = {
    "n't", "not", "never", "no", "without", "hardly", "barely",
    "fail", "unable", "unlikely", "absence", "prevent", "lack",
    "deny", "dismiss", "reject", "cannot"
}
sentiment_lemmatizer = WordNetLemmatizer()


def load_financial_dictionary_classifier():
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
        'warning', 'fail', 'bankruptcy', 'investigation', 'lawsuit', 'litigation', 'pressure', 'fear', 'recession',
        'bad',
        'concern', 'caution', 'downgrade', 'decrease', 'lose', 'challenge', 'bearish', 'degeneration', 'half',
        'downside'
    ]

    financial_neutral = [
        'unchanged', 'neutral', 'maintain', 'expect', 'keep'
                                                      'estimate', 'guidance', 'target', 'announce', 'report', 'quarter',
        'fiscal', 'year', 'slightly', 'slight', 'pretty', 'slow', 'prudent', 'insignificant', 'unsignificant'
    ]

    lexicon = {}
    for word in financial_positive:
        lexicon[sentiment_lemmatizer.lemmatize(word)] = random.uniform(1.5, 2)
    for word in financial_negative:
        lexicon[sentiment_lemmatizer.lemmatize(word)] = random.uniform(-2, -1.5)
    for word in financial_neutral:
        lexicon[sentiment_lemmatizer.lemmatize(word)] = 0

    return lexicon


financial_dictionary_classifier = load_financial_dictionary_classifier()


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()  # get first letter of POS tag
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_sentiment_text(text):
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # remove html
    text = re.sub(r'<.*?>', '', text)
    # tokenize
    tokens = word_tokenize(text)

    # lemmatize tokens and remove stopwords
    lemmatized_tokens = [sentiment_lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if
                         len(token) > 1]
    return ' '.join(lemmatized_tokens)


def load_financial_phrasebank():
    thread_safe_print("Loading Financial PhraseBank dataset...")
    data = []
    with open("fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r",
              encoding="ISO-8859-1") as file:
        for line in file:
            text, sentiment = line.rsplit("@", 1)
            sentiment = sentiment.strip()
            label = {"neutral": 0, "positive": 1, "negative": -1}.get(sentiment, 0)
            data.append({"text": text.strip(), "sentiment": label})

    dataset = pd.DataFrame(data)
    thread_safe_print(dataset['sentiment'].value_counts())
    return dataset


def train_sentiment_model():
    global sentiment_vectorizer, sentiment_model
    dataset = load_financial_phrasebank()
    dataset["processed_text"] = dataset["text"].apply(preprocess_sentiment_text)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["processed_text"], dataset["sentiment"], test_size=0.2, random_state=42
    )

    sentiment_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = sentiment_vectorizer.fit_transform(X_train)
    X_test_vec = sentiment_vectorizer.transform(X_test)

    sentiment_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    sentiment_model.fit(X_train_vec, y_train)

    # evaluate performance
    y_pred = sentiment_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    thread_safe_print(f"Model accuracy: {accuracy:.4f}")
    thread_safe_print(classification_report(y_test, y_pred))

    # save model
    with open("financial_sentiment_model.pkl", "wb") as f:
        pickle.dump((sentiment_vectorizer, sentiment_model), f)

    return accuracy


def load_sentiment_model():
    global sentiment_vectorizer, sentiment_model
    thread_safe_print("Loading sentiment model...")
    start_time = time.time()

    try:
        with open('financial_sentiment_model.pkl', 'rb') as f:
            sentiment_vectorizer, sentiment_model = pickle.load(f)
        thread_safe_print(f"Loaded pre-trained sentiment model")
    except FileNotFoundError:
        thread_safe_print("No pre-trained model found. Training a new model...")
        train_sentiment_model()
    except Exception as e:
        thread_safe_print(f"Error loading model: {e}")
        return False

    elapsed_time = time.time() - start_time
    thread_safe_print(f"Sentiment model loaded in {elapsed_time:.2f} seconds")
    return True


def fetch_news_for_ticker(ticker):
    thread_safe_print(f"Fetching news for {ticker}...")

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


def detect_negation_scope(text):
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
        if token in negation_words:
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
                # add meaningful words to negated set (not stopwords)
                if token not in stop_words:
                    # negated_words.add(token)
                    negated_words.add(sentiment_lemmatizer.lemmatize(token))

    return negated_words


def lexicon_score(text):
    text = text.lower()
    tokens = word_tokenize(text)
    score = 0

    phrase_sentiments = {
        "from buy to hold": -2, "from hold to sell": -2, "from buy to sell": -2,
        "downgrade to hold": -2, "downgrade to sell": -2, "cut to hold": -2, "cut to sell": -2,
        "downgrade": -2,
        "from sell to hold": 2, "from hold to buy": 2, "from sell to buy": 2, "reinstates dividend": 1.5,
        "profit boost": 1.5,
        "market leader": 2, "top performer": 2,
        "upgrade to buy": 2, "upgrade to hold": 2, "raise to buy": 2, "acquisition deal": 2, "strategic partnership": 2,
        "upgrade": 2, "new product": 1.5, "new offer": 1.5, "new offering": 1.5, "expansion plan": 1.5,
        "new contract": 1.5,
        "n't meet expectation": -2, "not profitable": -2, "no growth": -1.5,
        "n't achieve target": -2, "never recover": -1.5,
        "competitive pressure": -1.5, "margin concern": -1.5
    }

    for phrase, sentiment in phrase_sentiments.items():
        if phrase in text:
            score += sentiment

    negated_words = detect_negation_scope(text)

    for token in tokens:
        lemmatized_token = sentiment_lemmatizer.lemmatize(token)

        if lemmatized_token in financial_dictionary_classifier:
            sentiment = financial_dictionary_classifier[lemmatized_token]

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


def analyze_sentiment(text):
    global sentiment_vectorizer, sentiment_model
    if sentiment_model is None or sentiment_vectorizer is None:
        load_sentiment_model()

    title_content = text.split('. ', 1)
    title = title_content[0] if len(title_content) > 1 else ""

    # get processed text but preserve original for negation detection
    processed_text = preprocess_sentiment_text(text)

    # get prediction
    X_vec = sentiment_vectorizer.transform([processed_text])
    model_score = sentiment_model.predict_proba(X_vec)[0]

    # convert probability to score
    if len(model_score) == 3:
        ml_score = model_score[2] - model_score[0]  # pos_prob - neg_prob
    else:
        ml_score = model_score[1] * 2 - 1  # Convert 0-1 to -1 to 1

    negation_present = any(term in word_tokenize(processed_text) for term in negation_words)
    # for term in processed_text:
    #     if term in self.negation_words:
    #         print(term)
    if negation_present:
        ml_score *= 0.75

    lex_score = lexicon_score(processed_text)

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


# ======ARIMA PRICE PREDICTION MODEL======
arima_model_path = "E:\\saved_models\\arima_price_prediction_model"
arima_model_file = os.path.join(arima_model_path, "arima_model.pkl")


# Build and train ARIMA model
def build_and_train_model(training_data, validation_data):
    global arima_model
    thread_safe_print("Finding best ARIMA parameters with auto_arima")
    arima_model = auto_arima(training_data, seasonal=False, trace=True,
                             error_action='ignore', suppress_warnings=True)
    predictions = arima_model.predict(n_periods=len(validation_data))
    mse = mean_squared_error(validation_data, predictions)
    rmse = np.sqrt(mse)
    thread_safe_print(f"Validation RMSE: {rmse:.2f}")
    return predictions


# Predict next 12 months
def predict_next_12_months():
    global arima_model, stock_data, ticker
    if ticker is None:
        raise ValueError("Ticker symbol not set. Use set_ticker() method first.")
    if arima_model is None or stock_data is None:
        raise ValueError("Model has not been trained or data is not loaded")

    last_date = stock_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=252, freq='B')
    predictions = arima_model.predict(n_periods=252)

    # Introduce a small bias to integrate volatility
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i - 1]:
            predictions[i] *= 1.01
        else:
            predictions[i] *= 0.99

    future_predictions = pd.Series(predictions, index=future_dates)
    monthly_indices = pd.date_range(start=future_dates.min(), end=future_dates.max(), freq='ME')
    monthly_predictions = [future_predictions.loc[future_predictions.index[
        future_predictions.index.get_indexer([date], method='nearest')[0]]] for date in monthly_indices]

    return monthly_predictions, monthly_indices


# Save and load ARIMA model
def save_model():
    global arima_model
    if arima_model is None:
        raise ValueError("No model to save. Train a model first.")
    os.makedirs(arima_model_path, exist_ok=True)
    joblib.dump(arima_model, arima_model_file)
    thread_safe_print(f"Model saved to {arima_model_file}")


def load_arima_model():
    global arima_model
    thread_safe_print("Loading ARIMA model...")
    start_time = time.time()

    if os.path.exists(arima_model_file):
        arima_model = joblib.load(arima_model_file)
        thread_safe_print(f"Model loaded from {arima_model_file}")
        success = True
    else:
        thread_safe_print(f"No existing model found at {arima_model_file}")
        success = False

    elapsed_time = time.time() - start_time
    thread_safe_print(f"ARIMA model loading completed in {elapsed_time:.2f} seconds")
    return success


# Fetch stock data using yfinance
def fetch_stock_data(ticker_symbol, train_start='2010-01-01', train_end='2020-12-31',
                     val_start='2021-01-01', val_end='2023-01-01'):
    global stock_data, ticker
    ticker = ticker_symbol
    if ticker is None:
        raise ValueError("Ticker symbol not set. Provide a valid ticker.")

    import yfinance as yf
    stock_data = yf.download(ticker, start=train_start, end=val_end)
    stock_data = stock_data.asfreq('B').ffill().bfill()
    training_data = stock_data['Close'][train_start:train_end]
    validation_data = stock_data['Close'][val_start:val_end]
    return training_data, validation_data


# Run stock prediction
def run_stock_prediction(ticker_symbol):
    thread_safe_print(f"Processing {ticker_symbol} stock prediction")
    training_data, validation_data = fetch_stock_data(ticker_symbol)

    if load_arima_model():
        thread_safe_print("Updating model with all available data")
        all_data = stock_data['Close'].asfreq('B').ffill().bfill()
        arima_model.update(all_data)
    else:
        thread_safe_print("Training new ARIMA model")
        build_and_train_model(training_data, validation_data)
        save_model()

    thread_safe_print("Predicting prices for the next 12 months...")
    predictions, future_dates = predict_next_12_months()
    for i in range(len(predictions)):
        thread_safe_print(f"Price ${predictions[i]:.2f} at date {future_dates[i].strftime('%b %Y')}")


def adjust_predictions_with_sentiment(predictions, weighted_sentiment_score):
    adjusted_predictions = []
    score_intervals = {
        (-1, -0.75): (0.15, 0.2),
        (-0.75, -0.2): (0.1, 0.15),
        (-0.2, 0.2): (0.02, 0.1),
        (0.2, 0.75): (0.1, 0.15),
        (0.75, 1): (0.15, 0.2)
    }
    initial_impact_factor = 0.01

    for (low, high), impact in score_intervals.items():
        if low <= weighted_sentiment_score <= high:
            initial_impact_factor = random.uniform(impact[0], impact[1])

    for i, prediction in enumerate(predictions):
        # use exponential decay for a gradual decrease of the impact over the months
        gradual_impact = initial_impact_factor * (0.6 ** i)
        adjusted_prediction = prediction * (1 + weighted_sentiment_score * gradual_impact)
        adjusted_predictions.append(adjusted_prediction)

    return adjusted_predictions

def analyze_ticker_news(ticker, custom_article=None):
    # check if model trained/loaded
    if not os.path.exists('financial_sentiment_model.pkl'):
        print("Training new sentiment model...")
        train_sentiment_model()
    else:
        load_sentiment_model()

    if custom_article:
        result = analyze_sentiment(custom_article)
        print(f"\nAnalysis for custom article about {ticker}:")
        print(f"Sentiment: {result['sentiment_class']} (Score: {result['sentiment_score']})")
        print(f"Confidence: {result['model_confidence']:.2f}")
        return result

    articles = fetch_news_for_ticker(ticker)
    results = []

    print(f"\nSentiment Analysis for {ticker} news:")
    for i, article in enumerate(articles):
        result = analyze_sentiment(
            f"{article['title']}. {article['content']}",
            ticker
        )
        results.append(result)

        print(f"\nArticle {i + 1}:")
        print(f"Headline: {article['title']}")
        print(f"Sentiment: {result['sentiment_class']} (Score: {result['sentiment_score']})")

    avg_score = sum(r['sentiment_score'] for r in results) / len(results)
    print(f"\nOverall {ticker} sentiment: {avg_score:.3f}")

    return results



# if __name__ == "__main__":
#     news = "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"
#     ticker = "MSFT"
#     url = re.search(r'(https?://[^\s]+)$', news) or re.search(r'([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$', news)
#     load_credibility_model()
#     if url:
#         url = url.group(1)
#     else:
#         url = ''
#
#     if url == '':
#         clean_news = news
#     else:
#         clean_news = news.replace(url, '')
#
#     preprocessed_text = preprocess_credibility_text(clean_news)
#     transformed_news = credibility_vectorization.transform([preprocessed_text])
#
#     pred_value_score = credibility_model.predict_proba(transformed_news)[0][1]
#     source_credibility_score = check_source_credibility(url)
#     text_quality_score = text_quality_score(clean_news)
#     domain_legitimacy_score = domain_legitimacy(url)
#
#     credibility_score = (
#             pred_value_score * 0.3 +
#             source_credibility_score * 0.3 +
#             text_quality_score * 0.3 +
#             domain_legitimacy_score * 0.1
#     )
#     credibility_score = get_credibility(credibility_score)
#     print(f"cred: {credibility_score}")
#
#     load_sentiment_model()
#     analyze_ticker_news(ticker, news)





from concurrent.futures import ThreadPoolExecutor, as_completed
def load_all_models_in_parallel(ticker, train_start, train_end, val_start, val_end):
    load_functions = [
        load_credibility_model,
        load_sentiment_model,
        load_arima_model,
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(func): func.__name__ for func in load_functions}
        fetch_future = executor.submit(fetch_stock_data_parallel, ticker, train_start, train_end, val_start, val_end)
        futures[fetch_future] = "stock_data"

        for future in as_completed(futures):
            func_name = futures[future]
            try:
                result = future.result()
                thread_safe_print(f"{func_name} completed successfully.")
                if func_name == "stock_data":
                    stock_data = result
            except Exception as e:
                thread_safe_print(f"{func_name} generated an exception: {e}")

    return stock_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def calculate_pred_value_score(transformed_news):
    return credibility_model.predict_proba(transformed_news)[0][1]

def calculate_source_credibility_score(url):
    return check_source_credibility(url)

def calculate_text_quality_score(clean_news):
    return text_quality_score(clean_news)

def calculate_domain_legitimacy_score(url):
    return domain_legitimacy(url)

def run_credibility(news):
    url = re.search(r'(https?://[^\s]+)$', news) or re.search(r'([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$', news)
    url = url.group(1) if url else ''
    clean_news = news.replace(url, '') if url else news

    preprocessed_text = preprocess_credibility_text(clean_news)
    transformed_news = credibility_vectorization.transform([preprocessed_text])

    scores = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(calculate_pred_value_score, transformed_news): "pred_value_score",
            executor.submit(calculate_source_credibility_score, url): "source_credibility_score",
            executor.submit(calculate_text_quality_score, clean_news): "text_quality_score",
            executor.submit(calculate_domain_legitimacy_score, url): "domain_legitimacy_score"
        }

        for future in as_completed(futures):
            score_name = futures[future]
            try:
                scores[score_name] = future.result()
            except Exception as e:
                print(f"Error calculating {score_name}: {e}")
                scores[score_name] = 0  # 0 if an error occurs

    credibility_score = (
        scores.get("pred_value_score", 0) * 0.3 +
        scores.get("source_credibility_score", 0) * 0.3 +
        scores.get("text_quality_score", 0) * 0.3 +
        scores.get("domain_legitimacy_score", 0) * 0.1
    )

    # credibility_score = get_credibility(credibility_score)
    # print(f"cred: {credibility_score}")
    return credibility_score

def run_sentiment(news):
    return analyze_sentiment(news)

def run_arima(stock_data):
    all_data = stock_data['Close'].asfreq('B').ffill().bfill()
    arima_model.update(all_data)

    predictions, future_dates = predict_next_12_months()
    return (predictions, future_dates)

import yfinance as yf
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


if __name__ == "__main__":
    news = "Apple has announced they will increase the production of smartphones by 2025. https://www.reuters.com"
    ticker = "MSFT"
    train_start, train_end = "2010-01-01", "2018-01-01"
    val_start, val_end = "2019-01-02", "2022-01-01"
    stock_data = load_all_models_in_parallel(ticker, train_start, train_end, val_start, val_end)[2]
    print(stock_data.tail())

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_credibility, news): "credibility_score",
            executor.submit(run_sentiment, news): "sentiment_score",
            executor.submit(run_arima, stock_data): "arima_prediction"
        }
        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exception:
                print(f"Error calculating {key}: {exception}")
                results[key] = None
    credibility_score = results.get("credibility_score", 0)
    sentiment_score = results.get("sentiment_score", 0)["sentiment_score"]

    arima_results = results.get("arima_prediction", ([], []))
    print(arima_results)
    #predictions, future_dates = arima_results

    # print("ARIMA predicted prices for 12 months")
    # for i in range(len(predictions)):
    #     print(predictions[i])

    # credibility_score = run_credibility(news)
    # sentiment_score = run_sentiment(news)["sentiment_score"]
    # TODO: run sentiment model, then fetch data concurrently form yf, then run arima model, then try to run all of them together after loading all of them
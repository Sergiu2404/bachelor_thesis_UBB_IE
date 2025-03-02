import os
import re
import joblib
import zipfile
import validators
import pandas as pd
import language_tool_python
from nltk import word_tokenize
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class FinancialNewsCredibilityAnalyzer:
    def __init__(self, model_path="E:\\saved_models\\fake_news_detection_model\\credibility_model.zip"):
        self.model_path = model_path
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.vectorization = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self.CREDIBLE_FINANCIAL_SOURCES = {
            'wsj.com': 0.9, 'bloomberg.com': 0.9, 'ft.com': 0.9,
            'reuters.com': 0.9, 'cnbc.com': 0.85, 'economist.com': 0.85,
            'marketwatch.com': 0.85, 'barrons.com': 0.85, 'forbes.com': 0.85,
            'morningstar.com': 0.85, 'investors.com': 0.85, 'businessinsider.com': 0.8,
            'fool.com': 0.7, 'seekingalpha.com': 0.7, 'yahoo.com/finance': 0.8,
            'cnn.com': 0.85, 'nytimes.com': 0.85
        }
        self.SUSPICIOUS_DOMAINS = ['.com.co', '.co.com', '.lo', 'finance-news24',
                                   'breaking-finance', 'stockalert', 'investing-secrets',
                                   'financial-trends', 'money-news-now', 'wallst-alerts']
        self._load_or_train_model()

    def _load_or_train_model(self):
        if os.path.exists(self.model_path):
            with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                zip_ref.extractall("E:\\saved_models")

            self.model = joblib.load("E:\\saved_models\\fake_news_detection_model\\credibility_model.pkl")
            self.vectorization = joblib.load("E:\\saved_models\\fake_news_detection_model\\vectorizer.pkl")
            # self.feature_scaler = joblib.load("E:\\saved_models\\feature_scaler.pkl")
        else:
            self._train_model()
            self._save_model()

    def _train_model(self):
        try:
            true = pd.read_csv("./fake_news_datasets/fake_news_datasets/True.csv")
            fake = pd.read_csv("./fake_news_datasets/fake_news_datasets/Fake.csv")
        except Exception:
            true, fake = pd.DataFrame(columns=['text']), pd.DataFrame(columns=['text'])

        true['label'], fake['label'] = 1, 0
        news = pd.concat([true, fake], axis=0).drop(['title', 'subject', 'date'], axis=1, errors='ignore')
        news = news.sample(frac=1).reset_index(drop=True)
        news['text'] = news['text'].apply(self._preprocess_text)

        x_train, x_test, y_train, y_test = train_test_split(news['text'], news['label'], test_size=0.2, random_state=42)
        self.vectorization.fit(x_train)
        xv_train, xv_test = self.vectorization.transform(x_train), self.vectorization.transform(x_test)
        self.model.fit(xv_train, y_train)
        pred = self.model.predict_proba(xv_test)[:, 1]
        print(classification_report(y_test, (pred > 0.5).astype(int), zero_division=0))



    def _extract_urls_from_text(self, text):
        """Extract all URLs from text content"""
        if not isinstance(text, str):
            return []
        url_pattern = re.compile(r'https?://\S+|www\.\S+|[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\.[a-zA-Z]{2,}')
        return url_pattern.findall(text)

    def _save_model(self):
        os.makedirs("E:\\saved_models", exist_ok=True)

        joblib.dump(self.model, "E:\\saved_models\\fake_news_detection_model\\credibility_model.pkl")
        joblib.dump(self.vectorization, "E:\\saved_models\\fake_news_detection_model\\vectorizer.pkl")
        # joblib.dump(self.feature_scaler, "E:\\saved_models\\feature_scaler.pkl")

        with zipfile.ZipFile(self.model_path, 'w') as zip_ref:
            zip_ref.write("E:\\saved_models\\fake_news_detection_model\\credibility_model.pkl", "credibility_model.pkl")
            zip_ref.write("E:\\saved_models\\fake_news_detection_model\\vectorizer.pkl", "vectorizer.pkl")
            # zip_ref.write("E:\\saved_models\\feature_scaler.pkl", "feature_scaler.pkl")

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        return text.strip()

    def _check_source_credibility(self, url):
        if not url:
            return 0.8
        domain = urlparse(url).netloc.replace('www.', '')

        if domain in self.CREDIBLE_FINANCIAL_SOURCES:
            return self.CREDIBLE_FINANCIAL_SOURCES[domain]
        if any(suspicious_domain in domain for suspicious_domain in self.SUSPICIOUS_DOMAINS):
            return 0.2
        return 0.5

    def _grammatical_score(self, text):
        matches = self.language_tool.check(text)
        if len(matches) == 0:
            return 1

        for match in matches:
            print(match)

        return round(len(self.language_tool.check(text)) / max(len(word_tokenize(text)), 1), 3)

    def _text_quality_score(self, text):
        score = 1
        all_caps_patterns = re.findall(r'\b[A-Z]{3,}\b', text)

        all_caps_ratio = len(all_caps_patterns) / max(1, len(text.split()))
        excessive_punctuation_ratio = (text.count('!') + text.count('?') + text.count("  ")) / max(1, len(text))
        grammar_errors_ratio = self._grammatical_score(text)

        print(all_caps_ratio, excessive_punctuation_ratio, grammar_errors_ratio)
        if all_caps_ratio >= 0.2:
            score *= 0.5
        elif all_caps_ratio >= 0.1:
            score *= 0.7

        if excessive_punctuation_ratio >= 0.03:
            score *= 0.7
        elif excessive_punctuation_ratio >= 0.01:
            score *= 0.9

        # if grammar_errors_ratio > 0.12:
        #     score *= 0.5
        # elif grammar_errors_ratio > 0.07:
        #     score *= 0.7
        score = score * grammar_errors_ratio

        # quality_score = 1 - sum([all_caps_ratio >= 0.2, excessive_punctuation_ratio >= 0.02, grammar_errors_ratio > 0.09]) * 0.1
        print(score)
        return max(0.1, score)

    def _domain_legitimacy(self, url):
        score = 1
        if not url or not validators.url(url):
            score *= 0.4

        parsed_url = urlparse(url)
        protocol = parsed_url.scheme
        domain = parsed_url.netloc

        if protocol == "http":
            score *= 0.7  # reduce the score in case secure http is not used
        # urlparse(url).netloc
        for suspicious_domain in self.SUSPICIOUS_DOMAINS:
            if suspicious_domain in domain or domain in self.SUSPICIOUS_DOMAINS or len(domain) < 5:
                score *= 0.3  # reduce if domain is suspicious

        return score

    def get_credibility(self, credibility_score):
        credibility_texts = ['Highly Suspicious', 'Unreliable', 'Uncertain', 'Likely Credible', 'Highly Credible']
        credibility_index = int(credibility_score * 5)

        return {
            'credibility_score': credibility_score,
            'credibility': credibility_texts[credibility_index]
        }


    def analyze(self, news):
        url = re.search(r'(https?://[^\s]+)$', news) or re.search(r'([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$', news)
        if url:
            url = url.group(1)
        else:
            url = ''

        if url == '':
            clean_news = news
        else:
            clean_news = news.replace(url, '')

        preprocessed_text = self._preprocess_text(clean_news)
        transformed_news = self.vectorization.transform([preprocessed_text])

        pred_value_score = self.model.predict_proba(transformed_news)[0][1]
        source_credibility_score = self._check_source_credibility(url)
        text_quality_score = self._text_quality_score(clean_news)
        domain_legitimacy_score = self._domain_legitimacy(url)

        credibility_score = (
                pred_value_score * 0.3 +
                source_credibility_score * 0.3 +
                text_quality_score * 0.3 +
                domain_legitimacy_score * 0.1
        )
        return self.get_credibility(credibility_score)


if __name__ == "__main__":
    analyzer = FinancialNewsCredibilityAnalyzer()
    while True:
        news_article = input("Enter financial news text: ")
        print(analyzer.analyze(news_article)['credibility_score'], analyzer.analyze(news_article)['credibility'])
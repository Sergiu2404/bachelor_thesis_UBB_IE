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
            'marketwatch.com': 0.85, 'barrons.com': 0.85, 'forbes.com': 0.75,
            'morningstar.com': 0.75, 'investors.com': 0.75, 'businessinsider.com': 0.7,
            'fool.com': 0.7, 'seekingalpha.com': 0.65, 'yahoo.com/finance': 0.7,
            'cnn.com': 0.75, 'nytimes.com': 0.8
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

        if grammar_errors_ratio > 0.12:
            score *= 0.5
        elif grammar_errors_ratio > 0.07:
            score *= 0.7

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
        return f"Credibility: {credibility_score:.2f} ({['Highly Suspicious', 'Unreliable', 'Uncertain', 'Likely Credible', 'Highly Credible'][int(credibility_score * 5)]})"


if __name__ == "__main__":
    analyzer = FinancialNewsCredibilityAnalyzer()
    while True:
        print(analyzer.analyze(input("Enter financial news text: ")))




# import pandas as pd
# import numpy as np
# import re
# import validators
# from urllib.parse import urlparse
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# try:
#     true = pd.read_csv("./fake_news_datasets/fake_news_datasets/True.csv")
#     fake = pd.read_csv("./fake_news_datasets/fake_news_datasets/Fake.csv")
# except Exception as e:
#     print(f"Error loading datasets: {e}")
#
#     true = pd.DataFrame(columns=['text'])
#     fake = pd.DataFrame(columns=['text'])
#
# true['label'] = 1
# fake['label'] = 0
#
# news = pd.concat([true, fake], axis=0)
# news = news.drop(['title', 'subject', 'date'], axis=1, errors='ignore')
#
# news = news.sample(frac=1)  # reshuffle all the data
# news.reset_index(inplace=True)
# news.drop(['index'], axis=1, inplace=True)
#
#
# def preprocess_text(text) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\d', '', text)
#     text = re.sub(r'\n', '', text)
#     return text
#
#
# news['text'] = news['text'].apply(preprocess_text) # apply to each individual item in the column
#
# x = news['text']
# y = news['label']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#
# vectorization = TfidfVectorizer()
# xv_train = vectorization.fit_transform(x_train)
# xv_test = vectorization.transform(x_test)
#
#
# from sklearn.linear_model import LogisticRegression
#
# model = LogisticRegression(max_iter=1000)
# model.fit(xv_train, y_train)
#
# pred = model.predict_proba(xv_test)[:, 1]  # probability of being reliable news
#
# print(classification_report(y_test, (pred > 0.5).astype(int), zero_division=0))
#
#
# # credible financial news sources
# CREDIBLE_FINANCIAL_SOURCES = {
#     'wsj.com': 0.9,  # Wall Street Journal
#     'bloomberg.com': 0.9,  # Bloomberg
#     'ft.com': 0.9,  # Financial Times
#     'reuters.com': 0.85,  # Reuters
#     'cnbc.com': 0.8,  # CNBC
#     'economist.com': 0.85,  # The Economist
#     'marketwatch.com': 0.8,  # MarketWatch
#     'barrons.com': 0.8,  # Barron's
#     'forbes.com': 0.75,  # Forbes
#     'morningstar.com': 0.8,  # Morningstar
#     'investors.com': 0.75,  # Investor's Business Daily
#     'businessinsider.com': 0.7,  # Business Insider
#     'fool.com': 0.7,  # The Motley Fool
#     'seekingalpha.com': 0.65,  # Seeking Alpha
#     'yahoo.com/finance': 0.7,  # Yahoo Finance
#     'cnn.com/business': 0.75,  # CNN Business
#     'nytimes.com/business': 0.8  # New York Times Business
# }
#
# # suspicious domain patterns
# SUSPICIOUS_DOMAINS = [
#     '.com.co',
#     '.co.com',
#     '.lo',
#     'finance-news24',
#     'breaking-finance',
#     'stockalert',
#     'investing-secrets',
#     'financial-trends',
#     'money-news-now',
#     'wallst-alerts'
# ]
#
#
# import language_tool_python
# from nltk import word_tokenize
#
#
#
# def check_source_credibility(url):
#     """Check if the news source is from a credible financial outlet"""
#     if not url:
#         return 0.5  # neutral score if no URL provided
#
#     try:
#         domain = urlparse(url).netloc
#         if not domain:
#             domain = url  # in case only domain was provided
#
#         if domain.startswith('www.'):
#             domain = domain[4:]
#
#         # check exact matches
#         for credible_domain, score in CREDIBLE_FINANCIAL_SOURCES.items():
#             if credible_domain in domain:
#                 return score
#
#         # check suspicious domains
#         for suspicious in SUSPICIOUS_DOMAINS:
#             if suspicious in domain:
#                 return 0.2  # very low credibility for suspicious domains
#
#         return 0.5  # neutral score for unknown domains
#     except:
#         return 0.5  # default i neutral
#
#
# def get_gramatical_correctness_score(text):
#     language_grammatical_checker = language_tool_python.LanguageTool('en-US')
#     matches = language_grammatical_checker.check(text)  # finds all grammatical errors
#     for i, match in enumerate(matches):
#         print(f"Error {i}: {match}")
#
#     return round(len(matches) / len(word_tokenize(text)), 3)
#
# def check_text_quality(text):
#     """Evaluate text quality based on grammar, capitalization, and punctuation"""
#     if not isinstance(text, str):
#         return 0.5
#
#     # check for ALL CAPS or all_lower words (excluding common abbreviations)
#     all_caps_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
#     words = text.split()
#     all_caps_ratio = all_caps_count / len(words) if words else 0
#
#     # check for excessive punctuation
#     exclamation_count = text.count('!')
#     question_count = text.count('?')
#     excessive_punct_ratio = (exclamation_count + question_count) / len(text) if text else 0
#
#     # check for spelling (simple approach - count words not in a common word list)
#     # For a production system, you'd use a proper spell checker
#     gramatical_errors_ratio = get_gramatical_correctness_score(text)
#
#     # quality score (higher -> better)
#     quality_score = 1.0
#
#     if all_caps_ratio >= 0.2:
#         quality_score -= 0.2
#
#     if excessive_punct_ratio >= 0.02:
#         quality_score -= 0.2
#
#     if gramatical_errors_ratio > 0.09:  # grammatical issues
#         quality_score -= 0.2
#
#     return max(0.1, quality_score)
#
#
# def domain_legitimacy_check(url):
#     """Check if the URL appears legitimate"""
#     if not url:
#         return 0.5  # Neutral if no URL
#
#     # check if it's a valid URL
#     is_valid = validators.url(url) if validators else False
#
#     if not is_valid:
#         return 0.4  # reduce score for invalid URLs
#
#     try:
#         domain = urlparse(url).netloc
#
#         # check suspicious domains / patterns
#         for suspicious in SUSPICIOUS_DOMAINS:
#             if suspicious in domain:
#                 return 0.3
#
#         # check for too short domains (phishing)
#         if len(domain) < 5:
#             return 0.4
#
#         return 0.8  # legit
#     except:
#         return 0.5  # neutral
#
#
# def output_credibility(base_pred_value, source_url=None, article_text=""):
#     """Calculate final credibility score based on multiple factors"""
#     # base prediction from the model (0-1 range)
#     base_score = 1 / (1 + np.exp(-base_pred_value))  # sigmoid
#
#     # additional checks
#     source_score = check_source_credibility(source_url)
#     text_quality_score = check_text_quality(article_text)
#     domain_score = domain_legitimacy_check(source_url)
#
#     # weight different factors
#     final_score = (
#             base_score * 0.4 +  # model prediction (50%)
#             source_score * 0.25 +  # source credibility (25%)
#             text_quality_score * 0.25 +  # text quality (25%)
#             domain_score * 0.1  # domain credibility score (10%)
#     )
#
#     return final_score
#
#
# def extract_url(text):
#     """extract an URL from the end of the text"""
#     url_pattern = r'(https?://[^\s]+)$'
#     match = re.search(url_pattern, text)
#
#     if match:
#         return match.group(1)
#
#     # check for just domain at the end
#     domain_pattern = r'([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$'
#     match = re.search(domain_pattern, text)
#
#     if match:
#         return match.group(1)
#
#     return None
#
#
# def run_model(news):
#     url = extract_url(news)
#
#     if url: # if url found, remove it
#         clean_news = news.replace(url, '')
#     else:
#         clean_news = news
#
#     # prepare the news text for the model
#     testing_news = {"text": [clean_news]}
#     new_def_test = pd.DataFrame(testing_news)
#     new_def_test['text'] = new_def_test['text'].apply(preprocess_text)
#
#     new_x_test = new_def_test['text']
#     new_xv_test = vectorization.transform(new_x_test)
#
#     # get base prediction
#     pred_value = model.predict_proba(new_xv_test)[0][1]
#
#     # calculate the final credibility score with additional checks
#     credibility_score = output_credibility(
#         pred_value,
#         source_url=url,
#         article_text=clean_news
#     )
#
#     # prepare detailed report
#     report = f"Credibility Prediction: {credibility_score:.2f}\n"
#     report += f"ML Model Base Score: {pred_value:.2f}\n"
#
#     if url:
#         source_score = check_source_credibility(url)
#         domain_score = domain_legitimacy_check(url)
#         report += f"Source: {url}\n"
#         report += f"Source Credibility: {source_score:.2f}\n"
#         report += f"Domain Legitimacy: {domain_score:.2f}\n"
#     else:
#         report += "No source URL detected. Source credibility not assessed.\n"
#
#     text_quality = check_text_quality(clean_news)
#     report += f"Text Quality Score: {text_quality:.2f}\n"
#
#     # provide interpretation
#     if credibility_score > 0.8:
#         report += "Assessment: Highly credible financial news"
#     elif credibility_score > 0.6:
#         report += "Assessment: Likely credible financial news"
#     elif credibility_score > 0.4:
#         report += "Assessment: Uncertain credibility, verify with other sources"
#     elif credibility_score > 0.2:
#         report += "Assessment: Likely unreliable financial information"
#     else:
#         report += "Assessment: Highly suspicious content, likely fake news"
#
#     return report
#
#
# def train_with_new_example(news, is_credible=True):
#     """Train the model with a new example (online learning)"""
#     url = extract_url(news)
#
#     if url:
#         clean_news = news.replace(url, '')
#     else:
#         clean_news = news
#
#     label = 1 if is_credible else 0
#     new_data = pd.DataFrame({"text": [clean_news], "label": [label]})
#
#     new_data['text'] = new_data['text'].apply(preprocess_text)
#     new_x_new = new_data['text']
#     new_xv_new = vectorization.transform(new_x_new)
#
#     # Update the model (partial_fit would be better for online learning)
#     model.fit(new_xv_new, new_data['label'])
#     print("Model updated with new example.")
#
#
# if __name__ == "__main__":
#     print("Financial News Credibility Analyzer")
#     print("===================================")
#
#     while True:
#         news_input = input("\nEnter financial news text: ")
#
#         report = run_model(news_input)
#         print("\n--- CREDIBILITY REPORT ---")
#         print(report)
#         print("-------------------------")
#
#         feedback = input("\nIs this analysis correct? (y/n): ")
#         if feedback.lower() == 'n':
#             correct = input("Is this news credible? (y/n): ")
#             train_with_new_example(news_input, correct.lower() == 'y')
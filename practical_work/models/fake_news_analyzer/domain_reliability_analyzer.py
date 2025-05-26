import pandas as pd
import re
from datasets import load_dataset


class NewsReliabilityChecker:
    def __init__(self):
        self.reliability_dict = self._load_reliability_scores()

    def _load_reliability_scores(self):
        dataset = load_dataset('sergioburdisso/news_media_reliability')
        df = pd.DataFrame(dataset['train'])

        df = df.dropna(subset=['domain', 'newsguard_score'])
        df['score'] = df['newsguard_score'] / 100.0
        df['domain'] = df['domain'].str.lower()
        df = df.drop(columns=['newsguard_score'], errors='ignore')

        return dict(zip(df['domain'], df['score']))

    def extract_domain(self, text):
        pattern = r"https?://(?:www\.)?([^/\s]+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower()
        return None

    def get_reliability_score(self, text):
        domain = self.extract_domain(text)
        if domain and domain in self.reliability_dict:
            return self.reliability_dict[domain]
        return 0.5
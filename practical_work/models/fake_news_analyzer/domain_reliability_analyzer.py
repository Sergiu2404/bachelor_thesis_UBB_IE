import nltk
import pandas as pd
from datasets import load_dataset
import random
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

dataset = load_dataset('sergioburdisso/news_media_reliability')
df = pd.DataFrame(dataset['train'])

df = df.dropna(subset=['domain', 'newsguard_score'])

df['score'] = df['newsguard_score'] / 100.0
df['domain'] = df['domain'].str.lower()

print(df['score'].value_counts())
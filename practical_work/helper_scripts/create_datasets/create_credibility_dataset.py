# import kagglehub
#
# path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
#
# print("Path to dataset files:", path)

import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download("punkt")
nltk.download("stopwords")

import en_core_web_sm
nlp = en_core_web_sm.load()

stopwords = set(stopwords.words("english"))
df = pd.read_csv("E:\\welfake_dataset\\WELFake_Dataset.csv")
df = df[:int(0.5 * len(df))]
print(len(df))
df["text"] = df["title"].fillna('') + ' ' + df["text"].fillna('')
df["credibility_score"] = df["label"]
df.drop(columns=["Unnamed: 0", "title", "label"], inplace=True)
print(df.columns)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

print("started cleaning dataset")
df["text"] = df["text"].apply(clean_text)

print("saving to datasets")
df.to_csv("./credibility_datasets/36k_welfake_dataset.csv", index=False)
print("saved to csv")
print(df.columns)

file_path = "credibility_datasets/36k_welfake_dataset.csv"
file_size_bytes = os.path.getsize(file_path)
file_size_mb = file_size_bytes / (1024 * 1024)

print(f"Saved dataset to '{file_path}' ({file_size_mb:.2f} MB)")
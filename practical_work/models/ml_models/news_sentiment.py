# # pretrained finbert model for news sentiment analysis
#
# import feedparser
# from transformers import pipeline
#
# pipe = pipeline("text-classification", model="ProsusAI/finbert")
# #print(pipe("Stocks rallied and the British pound gained."))
#
# symbol = "META"
# keyword = "meta" #check for some word in title of the article
# rss_url = f'https://finance.yahoo.com/rss/headline?s={symbol}'
#
# feed = feedparser.parse(rss_url)
#
# total_score = 0
# num_articles = 0
#
# for i, entry in enumerate(feed.entries):
#     if keyword.lower() not in entry.summary.lower():
#         continue
#
#     print(f"title: {entry.title}")
#     print(f"link: {entry.link}")
#     print(f"published: {entry.published}")
#     print(f"summary: {entry.summary}")
#
#     sentiment = pipe(entry.summary)[0]
#     print(f"Sentiment: {sentiment['label']}, score: {sentiment['score']}")
#     print("-" * 20)
#
#     if sentiment['label'] == 'positive':
#         total_score += sentiment['score']
#         num_articles += 1
#     elif sentiment['label'] == 'negative':
#         total_score -= sentiment['score']
#         num_articles += 1
#
# final_score = total_score / num_articles
# print(f"Overall sentiment for news regarding {symbol} is {final_score}, so news is mostly {'positive' if final_score >= 0.15 else 'negative' if final_score <= -0.15 else 'neutral'}")





#
# import re
# import pandas as pd
# import numpy as np
# import nltk
# import glob
#
# file_path_pattern = "./fake_news_datasets/amazon-fine-reviews-dataset-splitted/Reviews*"
# all_files = glob.glob(file_path_pattern)
#
# def natural_sort(files):
#     return sorted(files, key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
#
# # Sort the files using natural sorting
# sorted_files = natural_sort(all_files)
#
# print(sorted_files)
#
# df = pd.concat([pd.read_csv(f) for f in sorted_files], axis=0)
# print(df.tail())
#
# # df = pd.read_csv("/amazon-fine-reviews-dataset-splitted/Reviews1.csv")
# # df.head()
#
# example = df["Text"].iloc[50]
# print(example)
#
# # nltk.download('punkt_tab')
# tokens = nltk.word_tokenize(example)
# print(tokens[:10])
#
# #nltk.download('averaged_perceptron_tagger_eng')
# #
# # #get the tokens and their grammatical categories
# tagged = nltk.pos_tag(tokens)
# print(tagged[:10])
#
# #nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
# entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()
#
# from tqdm import tqdm
#
# for _, row in tqdm(df.iterrows(), total=len(df)):
#   text = row["Text"]
#
# from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
#
# sia = SentimentIntensityAnalyzer()
# print(sia.polarity_scores("I am very well"))






import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)



from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Add financial terms with stronger sentiment weights
financial_lexicon = {
    "bullish": 2.5,
    "bearish": -2.5,
    "upgrade": 1.5,
    "downgrade": -1.5,
    "profit": 2.0,
    "loss": -2.0,
    "rally": 1.8,
    "plunge": -2.2
}

sia.lexicon.update(financial_lexicon)

example_texts = [
    "The stock market experienced a bullish rally today.",
    "Company reports a significant loss in Q4 earnings.",
    "Analyst upgrades the stock to a strong buy."
]

for text in example_texts:
    print(text, "->", sia.polarity_scores(text))

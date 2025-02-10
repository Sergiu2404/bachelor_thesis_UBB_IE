import pandas as pd
import numpy as np
import re

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

true = pd.read_csv("./fake_news_datasets/News _dataset/True.csv")
fake = pd.read_csv("./fake_news_datasets/News _dataset/Fake.csv")

true['label'] = 1
fake['label'] = 0

news = pd.concat([true, fake], axis=0)
news = news.drop(['title', 'subject', 'date'], axis=1)

news = news.sample(frac=1)  # reshuffle all the data
news.reset_index(inplace=True)  # reset the indexes
news.drop(['index'], axis=1, inplace=True)  # drop the index column

def preprocess_text(text) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', '', text)
    return text

news['text'] = news['text'].apply(preprocess_text)

x = news['text']
y = news['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LinearRegression

# Linear Regression model for continuous output
lin_reg = LinearRegression()
lin_reg.fit(xv_train, y_train)

pred_lin_reg = lin_reg.predict(xv_test)

print(classification_report(y_test, (pred_lin_reg > 0.5).astype(int), zero_division=0))  # convert to binary for evaluation

def output_credibility(pred_value):
    return 1 / (1 + np.exp(-pred_value))  # apply sigmoid function for probability scaling

# def run_model(news):
#     testing_news = {"text": [news]}
#     new_def_test = pd.DataFrame(testing_news)
#     new_def_test['text'] = new_def_test['text'].apply(preprocess_text)
#
#     new_x_test = new_def_test['text']
#     new_xv_test = vectorization.transform(new_x_test)
#
#     pred_lin_reg = lin_reg.predict(new_xv_test)
#     credibility_score = output_credibility(pred_lin_reg[0])
#     return f"Linear Regression Credibility Prediction: {credibility_score:.2f}"

def run_model(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(preprocess_text)

    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)

    pred_lin_reg = lin_reg.predict(new_xv_test)
    credibility_score = output_credibility(pred_lin_reg[0])
    return f"Linear Regression Credibility Prediction: {credibility_score:.2f}"

    new_data = pd.DataFrame({"text": [news], "label": [1]})

    # Preprocess the new data
    new_data['text'] = new_data['text'].apply(preprocess_text)
    new_x_new = new_data['text']
    new_xv_new = vectorization.transform(new_x_new)

    # Retrain the model incrementally (online learning)
    lin_reg.fit(new_xv_new, new_data['label'])
    print("Model updated with new data.")

while True:
    news_input = input("Give some news as input: ")
    print(run_model(news_input))

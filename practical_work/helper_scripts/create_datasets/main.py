import pandas as pd

df = pd.read_csv("./fake_datasets/Fake.csv")
print(df.shape[0])
print(df.size)

import spacy

intensifiers = [
    "n't", "much", "major", "significant", "substantial", "considerable", "notable",
    "great", "large", "strong", "tremendous", "enormous", "massive", "extensive",
    "marked", "pronounced", "immense", "meaningful", "huge", "hefty"
]

nlp = spacy.load("en_core_web_sm")

# Lemmatize and filter out stop words
lemmatized = [
    token.lemma_
    for word in intensifiers
    for token in nlp(word)
    if not token.is_stop
]

print("Lemmatized list without stop words:", lemmatized)

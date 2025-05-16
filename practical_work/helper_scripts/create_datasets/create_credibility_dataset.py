import pandas as pd
import re
import nltk
import os
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

import en_core_web_sm
nlp = en_core_web_sm.load()


df = pd.read_csv("E:\\welfake_dataset\\WELFake_Dataset.csv")
df["text"] = df["title"].fillna('') + ' ' + df["text"].fillna('')

df["credibility_score"] = df["label"].apply(lambda x: 0 if x == 1 else 1)

df.drop(columns=["Unnamed: 0", "title", "label"], inplace=True)

# def clean_text(text):
#     text = re.sub(r'[^a-zA-Z\s]', '', str(text))
#     text = text.lower()
#     tokens = word_tokenize(text)
#     doc = nlp(" ".join(tokens))
#     lemmatized = [token.lemma_ for token in doc]
#     return " ".join(lemmatized)

# print("Started cleaning dataset...")
# df["text"] = df["text"].apply(clean_text)
# print("Finished cleaning dataset.")

output_dir = "./credibility_datasets"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 10000
total_records = len(df)
for i in range(0, total_records, chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_num = (i // chunk_size) + 1
    file_path = os.path.join(output_dir, f"welfake_chunk_{chunk_num}.csv")
    chunk.to_csv(file_path, index=False)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Saved chunk {chunk_num} to '{file_path}' ({file_size_mb:.2f} MB)")

print("All chunks saved.")





# import os
# import pandas as pd
# import kagglehub
#
# # Download dataset using kagglehub
# path = kagglehub.dataset_download("abaghyangor/fake-news-dataset")
# print("Path to dataset files:", path)
#
# # Load the main CSV file from the downloaded path
# # Adjust the filename if necessary (you may need to check the actual file name in the directory)
# dataset_file = os.path.join(path, "fake-news-dataset.csv")
# df = pd.read_csv(dataset_file)
#
# # Target output directory
# output_dir = "E:\\datasets\\fake_news_detection"
# os.makedirs(output_dir, exist_ok=True)
#
# # Define base name for saved files
# base_name = "fake_news_abaghyan"
#
# # Split and save the dataset into chunks of 10k records
# chunk_size = 10000
# total_records = len(df)
#
# for i in range(0, total_records, chunk_size):
#     chunk = df.iloc[i:i+chunk_size]
#     chunk_num = (i // chunk_size) + 1
#     file_path = os.path.join(output_dir, f"{base_name}_chunk_{chunk_num}.csv")
#     chunk.to_csv(file_path, index=False)
#     print(f"Saved chunk {chunk_num} to '{file_path}' ({len(chunk)} records)")
#
# print("All chunks saved.")

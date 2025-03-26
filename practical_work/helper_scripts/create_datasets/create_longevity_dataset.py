# import os
# import pandas as pd
# from tqdm import tqdm
# import random
# import torch
# from transformers import pipeline
#
# def load_fiqa_dataset():
#     print("Loading FiQA dataset...")
#     splits = {
#         'train': 'data/train-00000-of-00001-aeefa1eadf5be10b.parquet'
#     }
#
#     df_fiqa = pd.read_parquet("hf://datasets/TheFinAI/fiqa-sentiment-classification/" + splits["train"])
#     df_fiqa = df_fiqa[['sentence', 'score']].rename(columns={'sentence': 'text', 'score': 'sentiment'})
#
#     return df_fiqa
#
# def setup_zero_shot_classifier():
#     print("Setting up zero-shot classifier...")
#     model_name = "facebook/bart-large-mnli"
#
#     classifier = pipeline(
#         "zero-shot-classification",
#         model=model_name,
#         device=0 if torch.cuda.is_available() else -1
#     )
#     return classifier
#
# def generate_longevity_dataset_with_zero_shot(df, classifier, sample_size=None, batch_size=16):
#     if sample_size is not None:
#         df = df.sample(sample_size, random_state=42)
#
#     texts = df['text'].tolist()
#     print(f"Processing {len(texts)} texts with zero-shot classification...")
#
#     candidate_labels = [
#         "very short-term financial impact (days to weeks)",
#         "short-term financial impact (weeks to months)",
#         "medium-term financial impact (months to a year)",
#         "long-term financial impact (1-3 years)",
#         "very long-term financial impact (3+ years)"
#     ]
#
#     label_values = {
#         "very short-term financial impact (days to weeks)": 0.1,
#         "short-term financial impact (weeks to months)": 0.3,
#         "medium-term financial impact (months to a year)": 0.5,
#         "long-term financial impact (1-3 years)": 0.7,
#         "very long-term financial impact (3+ years)": 0.9
#     }
#
#     all_scores = []
#
#     for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
#         batch = texts[i:i + batch_size]
#         batch_results = []
#
#         for text in batch:
#             try:
#                 result = classifier(text, candidate_labels)
#                 weighted_score = sum(score * label_values[label] for label, score in zip(result['labels'], result['scores']))
#                 batch_results.append(weighted_score)
#             except Exception as e:
#                 print(f"Error processing text: {e}")
#                 batch_results.append(random.uniform(0.4, 0.6))
#
#         all_scores.extend(batch_results)
#
#     longevity_df = pd.DataFrame({
#         'text': texts,
#         'longevity': all_scores
#     })
#
#     return longevity_df
#
# def create_longevity_dataset(sample_size=None):
#     os.makedirs("./sentiment_datasets", exist_ok=True)
#
#     df_fiqa = load_fiqa_dataset()
#     classifier = setup_zero_shot_classifier()
#     longevity_df = generate_longevity_dataset_with_zero_shot(df_fiqa, classifier, sample_size)
#
#     output_path = "./sentiment_datasets/longevity_dataset.csv"
#     longevity_df.to_csv(output_path, index=False)
#     print(f"Dataset saved to {output_path}")
#
#     longevity_df.to_parquet("./sentiment_datasets/longevity_dataset.parquet", index=False)
#     print(f"Dataset also saved as parquet to ./sentiment_datasets/longevity_dataset.parquet")
#
#     return longevity_df
#
# if __name__ == "__main__":
#     df = pd.read_parquet("./sentiment_datasets/longevity_dataset.parquet")
#     print(df.head())



   #  import argparse
   #
   #  parser = argparse.ArgumentParser(description='Generate a text longevity dataset from FiQA')
   #  parser.add_argument('--sample', type=int, default=None, help='Number of samples to process (for testing)')
   #  args = parser.parse_args()
   #
   #  if args.sample:
   #      print(f"Processing sample of {args.sample} texts")
   #
   #  df = create_longevity_dataset(sample_size=args.sample)
   #
   #  print("\nDataset statistics:")
   #  print(f"Total samples: {len(df)}")
   #  print(f"Longevity score distribution:")
   #  print(f"  Min: {df['longevity'].min():.2f}")
   #  print(f"  Max: {df['longevity'].max():.2f}")
   #  print(f"  Mean: {df['longevity'].mean():.2f}")
   #  print(f"  Median: {df['longevity'].median():.2f}")
   #
   #  bins = [0, 0.25, 0.5, 0.75, 1.0]
   #  labels = ['Very short-term', 'Short-term', 'Long-term', 'Very long-term']
   #  df['category'] = pd.cut(df['longevity'], bins=bins, labels=labels)
   #  distribution = df['category'].value_counts(normalize=True) * 100
   #
   #  print("\nDistribution by category:")
   #  for category, percentage in distribution.items():
   #      print(f"  {category}: {percentage:.1f}%")
   #



#APPEND SOME NEW LONGEVIVE RECORDS
# import pandas as pd
# import random
# from transformers import pipeline
# import kagglehub
#
# def load_kaggle_dataset():
#     """Load and preprocess the Kaggle Sentiment Analysis for Financial News dataset."""
#     print("Loading Kaggle Financial News dataset...")
#     path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
#     kaggle_df = pd.read_csv(f"{path}/all-data.csv", encoding="ISO-8859-1", header=None)
#
#     kaggle_df.columns = ["sentiment", "text"]
#     sentiment_mapping = {"negative": 2, "neutral": 0, "positive": 1}
#     kaggle_df["sentiment"] = kaggle_df["sentiment"].map(sentiment_mapping)
#
#     print("Kaggle dataset loaded:", kaggle_df.shape)
#     return kaggle_df
#
# news_categories = [
#     ("impact over the next two years", random.uniform(0.6, 0.7)),
#     ("impact over the next decade", random.uniform(0.7, 0.8)),
#     ("impact on future generations", random.uniform(0.8, 0.9))
# ]
#
# generator = pipeline("text-generation", model="ProsusAI/finbert")
#
# new_records = []
# for _ in range(180):
#     category, score = random.choice(news_categories)
#     prompt = f"Generate a financial news headline that describes an event with {category}. Different than the ones before"
#     response = generator(prompt, max_length=30, do_sample=True, top_k=50, temperature=0.7)
#     generated_text = response[0]['generated_text'].strip()
#     new_records.append((generated_text, score))
#
# df = pd.read_csv("./sentiment_datasets/longevity_dataset.csv")
# new_df = pd.DataFrame(new_records, columns=["text", "longevity"])
# df = pd.concat([df, new_df]).sample(frac=1, random_state=42).reset_index(drop=True)
# df.to_csv("./sentiment_datasets/longevity_dataset.csv", index=False)





#append kaggle to longevity and classify kaggle text for longevity
import pandas as pd
import numpy as np
import random
import torch
from transformers import pipeline
from kagglehub import dataset_download

print("Loading FinBERT model...")
finbert = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)


def load_kaggle_dataset():
    print("Loading Kaggle Financial News dataset...")
    path = dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    kaggle_df = pd.read_csv(f"{path}/all-data.csv", encoding="ISO-8859-1", header=None)

    kaggle_df.columns = ["sentiment", "text"]
    print("Kaggle dataset loaded:", kaggle_df.shape)
    return kaggle_df


def classify_longevity(text):
    result = finbert(text)

    neg_score = result[0][0]['score']
    neu_score = result[0][1]['score']
    pos_score = result[0][2]['score']

    if pos_score > 0.6:
        longevity = random.uniform(0.8, 0.9)
    elif neu_score > 0.6:
        longevity = random.uniform(0.4, 0.6)
    else:
        longevity = random.uniform(0.1, 0.3)

    return longevity


df_kaggle = load_kaggle_dataset()
df_existing = pd.read_csv("sentiment_datasets/longevity_dataset.csv")

new_records = []
short_term_count = 0
long_term_count = 0

for text in df_kaggle["text"]:
    longevity_score = classify_longevity(text)

    if longevity_score < 0.3:
        short_term_count += 1
        if short_term_count > long_term_count:
            text += " The long-term economic outlook remains promising."
            longevity_score = random.uniform(0.7, 0.9)

    if longevity_score > 0.7:
        long_term_count += 1

    new_records.append((text, longevity_score))

additional_records = [
    (
    "With a strategic vision spanning the next two decades, the firm is doubling down on AI research to future-proof its product offerings.",
    0.7456657803058624),
    (
    "@catskillfishing  Note the monthly chart of $GDX & how it recaptured support. $GLD possible double bottom set up. http://stks.co/aw6z",
    0.4119958251714706)
]
new_records.extend(additional_records)

df_new = pd.DataFrame(new_records, columns=["text", "longevity"])

df_final = pd.concat([df_existing, df_new]).sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv("./sentiment_datasets/longevity_dataset.csv", index=False)
print("New augmented dataset saved successfully!")

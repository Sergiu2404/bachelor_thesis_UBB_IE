import kagglehub
import yfinance as yf
import concurrent.futures
import time
from datetime import datetime, timedelta
import pandas as pd


def fetch_stock_data_for_range(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.asfreq('B')
    data = data.ffill().bfill()
    return data


def split_date_range(start_date, end_date, chunk_days=730):  # 730 days -> 2 years (one thr for 2 years)
    date_ranges = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        date_ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)  # next range starts form next day

    return date_ranges


def fetch_stock_data_parallel(ticker, start_date, end_date):
    date_ranges = split_date_range(start_date, end_date)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_stock_data_for_range, ticker, start, end) for start, end in date_ranges]

        all_data = []
        for future in concurrent.futures.as_completed(futures):
            all_data.append(future.result())

    combined_data = pd.concat(all_data)
    return combined_data


#if __name__ == '__main__':
#     ticker = "AAPL"
#     start_date = datetime(1970, 1, 1)
#     end_date = datetime(2025, 1, 1)
#
#     start_time = time.time()
#     data = fetch_stock_data_for_range(ticker, start_date, end_date)
#     print(f"fetching data in a single thread took: {time.time() - start_time} seconds")
#
#     start_parallel_time = time.time()
#     parallel_data = fetch_stock_data_parallel(ticker, start_date, end_date)
#     print(f"fetching data concurrently took: {time.time() - start_parallel_time} seconds")
#
#     print(parallel_data.head())




# import pandas as pd
#
# def load_fiqa_dataset():
#     """Load and preprocess the FiQA dataset."""
#     print("Loading FiQA dataset...")
#     splits = {
#         'train': 'data/train-00000-of-00001-aeefa1eadf5be10b.parquet',
#         'test': 'data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet',
#         'valid': 'data/valid-00000-of-00001-51867fe1ac59af78.parquet'
#     }
#
#     df_fiqa = pd.read_parquet("hf://datasets/TheFinAI/fiqa-sentiment-classification/" + splits["train"])
#
#     # Select relevant columns and rename
#     df_fiqa = df_fiqa[['sentence', 'score']].rename(columns={'sentence': 'text', 'score': 'sentiment'})
#
#     # Filter rows where sentiment score is between -0.1 and 0.1
#     neutral = df_fiqa[(df_fiqa['sentiment'] > -0.15) & (df_fiqa['sentiment'] < 0.15)]
#     positive = df_fiqa[(df_fiqa['sentiment'] > 0.15)]
#     negative = df_fiqa[(df_fiqa['sentiment'] < -0.15)]
#
#     print("FiQA dataset loaded:", df_fiqa.shape)
#     return (neutral, positive, negative)
#
# print(f"neutral: {load_fiqa_dataset()[0].shape[0]}")
# print(load_fiqa_dataset()[0])
# print(f"positive: {load_fiqa_dataset()[1].shape[0]}")
# print(load_fiqa_dataset()[1])
# print(f"negative: {load_fiqa_dataset()[2].shape[0]}")
# print(load_fiqa_dataset()[2])


import pandas as pd

def load_kaggle_dataset():
    """Load and preprocess the Kaggle Sentiment Analysis for Financial News dataset."""
    print("Loading Kaggle Financial News dataset...")
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    kaggle_df = pd.read_csv(f"{path}/all-data.csv", encoding="ISO-8859-1", header=None)

    kaggle_df.columns = ["sentiment", "text"]
    sentiment_mapping = {"negative": 2, "neutral": 0, "positive": 1}
    kaggle_df["sentiment"] = kaggle_df["sentiment"].map(sentiment_mapping)

    print("Kaggle dataset loaded:", kaggle_df.shape)

    # Display only the text and its sentiment for the first 2 rows of each sentiment category (0, 1, 2)
    for sentiment in [0, 1, 2]:
        print(f"\nSentiment {sentiment}:")
        rows = kaggle_df[kaggle_df['sentiment'] == sentiment].head(2)
        for _, row in rows.iterrows():
            print(f"Sentiment {row['sentiment']}: {row['text']}")

    return kaggle_df

print(load_kaggle_dataset())

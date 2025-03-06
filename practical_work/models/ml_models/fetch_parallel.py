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


if __name__ == '__main__':
    ticker = "AAPL"
    start_date = datetime(1970, 1, 1)
    end_date = datetime(2025, 1, 1)

    start_time = time.time()
    data = fetch_stock_data_for_range(ticker, start_date, end_date)
    print(f"fetching data in a single thread took: {time.time() - start_time} seconds")

    start_parallel_time = time.time()
    parallel_data = fetch_stock_data_parallel(ticker, start_date, end_date)
    print(f"fetching data concurrently took: {time.time() - start_parallel_time} seconds")

    print(parallel_data.head())
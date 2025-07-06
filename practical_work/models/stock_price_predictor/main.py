#
# gspc = yf.download("^GSPC", start="2025-05-24", end="2025-06-12")
#
# print(gspc)


import yfinance as yf
def save_localy_data_to_csv(ticker):
    df = yf.download(ticker, period="max")
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df = df[['Close', 'Volume', 'Volatility']].dropna()
    df.to_csv(fr"E:\thesis_fallback_datasets\{ticker}_processed.csv")

save_localy_data_to_csv("^GSPC")
save_localy_data_to_csv("AAPL")
save_localy_data_to_csv("GOOGL")
save_localy_data_to_csv("NVDA")
save_localy_data_to_csv("MSFT")
save_localy_data_to_csv("TSLA")
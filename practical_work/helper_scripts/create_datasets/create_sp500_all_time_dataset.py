from curl_cffi import requests

try:
    session = requests.Session(impersonate="chrome")
    response = session.get("https://www.google.com")
    print("Success:", response.status_code)
except Exception as e:
    print("Failed impersonation:", e)


import yfinance as yf
from curl_cffi import requests

session = requests.Session(impersonate="chrome")

ticker = yf.Ticker("^GSPC", session=session)
data = ticker.history(period="max")

data.to_csv("./stock_prices_datasets/sp500_all_time.csv")
print(data.head())

from typing import List
from data.services.stock_data.stock_provider_interface import StockDataProvider
import yfinance as yf
from yahooquery import Ticker
import pandas as pd
from curl_cffi import requests


class YahooFinanceProvider(StockDataProvider):
    def __init__(self):
        print("Initializing YahooFinanceProvider with curl_cffi session")
        self.session = requests.Session(impersonate="chrome")

    def get_stock_data(self, symbol: str) -> dict:

        try:
            print(f"Trying yfinance.download for {symbol}")
            df = yf.download(symbol, period="1d")
            if not df.empty:
                price = df["Close"].iloc[-1]
                info = yf.Ticker(symbol).info
                return {
                    "provider": "yfinance",
                    "company_name": info.get("longName", "N/A"),
                    "symbol": symbol,
                    "latest_price": price
                }
        except Exception as e:
            print(f"yfinance.download failed: {e}")


        try:
            print(f"Trying yahooquery for {symbol}")
            ticker = Ticker(symbol, session=self.session)
            info = ticker.price.get(symbol)
            if info and "regularMarketPrice" in info:
                return {
                    "provider": "yahooquery",
                    "company_name": info.get("longName", "N/A"),
                    "symbol": symbol,
                    "latest_price": info["regularMarketPrice"]
                }
        except Exception as e:
            print(f"[ERROR] yahooquery failed: {e}")

        return {"error": f"Could not retrieve stock data for {symbol}"}

    def get_monthly_close_prices(self, symbol: str):
        try:
            print(f"Trying yahooquery for monthly data of {symbol}")
            ticker = Ticker(symbol, session=self.session)
            history = ticker.history(period="5y", interval="1mo")
            if not history.empty:
                history.reset_index(inplace=True)
                history["date"] = pd.to_datetime(history["date"], errors='coerce')
                history["Year-Month"] = history["date"].dt.strftime("%Y-%m")
                monthly_data = history.groupby("Year-Month")["close"].last().to_dict()
                return {
                    "provider": "yahooquery",
                    "symbol": symbol,
                    "monthly_prices": {k: {"Close": v} for k, v in monthly_data.items()}
                }
        except Exception as e:
            print(f"[ERROR] yahooquery monthly failed: {e}")

        try:
            print(f"Trying yfinance.download for monthly data of {symbol}")
            df = yf.download(symbol, period="5y", interval="1mo")
            if not df.empty:
                df.reset_index(inplace=True)
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df["Year-Month"] = df["Date"].dt.strftime("%Y-%m")
                monthly_data = df.groupby("Year-Month")["Close"].last().to_dict()
                return {
                    "provider": "yfinance",
                    "symbol": symbol,
                    "monthly_prices": {k: {"Close": v} for k, v in monthly_data.items()}
                }
        except Exception as e:
            print(f"yfinance.download failed for monthly: {e}")

        return {"error": f"Could not retrieve monthly data for {symbol}"}

    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        try:
            print(f"Searching for symbols matching: {symbol_substr}")
            url = (
                f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol_substr}"
                f"&quotes_count=10&news_count=0&lang=en-US&region=US&corsDomain=finance.yahoo.com"
            )
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            resp = self.session.get(url, headers=headers)

            if resp.status_code != 200:
                return [{"error": f"Yahoo search failed with status code {resp.status_code}"}]

            results = resp.json()
            if "quotes" not in results:
                return [{"error": "No matching stocks found"}]

            symbols = [q.get("symbol") for q in results["quotes"] if q.get("symbol")]
            tickers = Ticker(symbols, session=self.session)
            prices = tickers.price

            return [
                {
                    "provider": "yahooquery",
                    "company_name": q.get("shortname", "N/A"),
                    "symbol": q.get("symbol", "N/A"),
                    "latest_price": prices.get(q["symbol"], {}).get("regularMarketPrice", "N/A")
                }
                for q in results["quotes"]
            ]

        except Exception as e:
            print(f"[ERROR] Exception in get_stocks_data_for_symbol_substring: {e}")
            return [{"error": f"Failed to retrieve symbols for '{symbol_substr}'"}]

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
                price = round(df["Close"].iloc[-1].item(), 2)
                print(price)
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
                    "latest_price": info.get("regularMarketPrice")
                }
        except Exception as e:
            print(f"[ERROR] yahooquery failed: {e}")

        return {"error": f"Could not retrieve stock data for {symbol}"}

    # def get_monthly_close_prices(self, symbol: str):
    #
    #     try:
    #         print(f"Trying yfinance.download for monthly data of {symbol}")
    #         df = yf.download(symbol, period="5y", interval="1mo")
    #         if not df.empty:
    #             df.reset_index(inplace=True)
    #             df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    #             df["Year-Month"] = df["Date"].dt.strftime("%Y-%m")
    #             monthly_data = df.groupby("Year-Month")["Close"].last().to_dict()
    #             return {
    #                 "provider": "yfinance",
    #                 "symbol": symbol,
    #                 "monthly_prices": {k: {"Close": v} for k, v in monthly_data.items()}
    #             }
    #     except Exception as e:
    #         print(f"yfinance.download failed for monthly: {e}")
    #
    #     try:
    #         print(f"Trying yahooquery for monthly data of {symbol}")
    #         ticker = Ticker(symbol, session=self.session)
    #         history = ticker.history(period="5y", interval="1mo")
    #         if not history.empty:
    #             history.reset_index(inplace=True)
    #             history["date"] = pd.to_datetime(history["date"], errors='coerce')
    #             history["Year-Month"] = history["date"].dt.strftime("%Y-%m")
    #             monthly_data = history.groupby("Year-Month")["close"].last().to_dict()
    #             return {
    #                 "provider": "yahooquery",
    #                 "symbol": symbol,
    #                 "monthly_prices": {k: {"Close": v} for k, v in monthly_data.items()}
    #             }
    #     except Exception as e:
    #         print(f"[ERROR] yahooquery monthly failed: {e}")
    #
    #     return {"error": f"Could not retrieve monthly data for {symbol}"}

    def get_monthly_close_prices(self, symbol: str):
        try:
            print(f"Trying yfinance.download for monthly data of {symbol}")
            df = yf.download(symbol, period="5y", interval="1mo")
            if not df.empty:
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                df.reset_index(inplace=True)
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

                df = df.dropna(subset=['Date'])

                df["Year-Month"] = df["Date"].dt.strftime("%Y-%m")

                monthly_data = df.groupby("Year-Month")["Close"].last()
                monthly_dict = {}

                for year_month, close_price in monthly_data.items():
                    # Convert numpy types to Python native types
                    if pd.isna(close_price):
                        close_price = None
                    else:
                        close_price = float(close_price)

                    monthly_dict[str(year_month)] = {"Close": round(close_price, 2)}

                return {
                    "provider": "yfinance",
                    "symbol": symbol,
                    "monthly_prices": monthly_dict
                }
        except Exception as e:
            print(f"yfinance.download failed for monthly: {e}")

        try:
            print(f"Trying yahooquery for monthly data of {symbol}")
            ticker = Ticker(symbol, session=self.session)
            history = ticker.history(period="5y", interval="1mo")
            if not history.empty:
                history.reset_index(inplace=True)
                history["date"] = pd.to_datetime(history["date"], errors='coerce')

                # Remove any rows with NaT dates
                history = history.dropna(subset=['date'])

                history["Year-Month"] = history["date"].dt.strftime("%Y-%m")

                # Convert to dict and ensure all values are JSON serializable
                monthly_data = history.groupby("Year-Month")["close"].last()
                monthly_dict = {}

                for year_month, close_price in monthly_data.items():
                    # Convert numpy types to Python native types
                    if pd.isna(close_price):
                        close_price = None
                    else:
                        close_price = float(close_price)

                    monthly_dict[str(year_month)] = {"Close": close_price}

                return {
                    "provider": "yahooquery",
                    "symbol": symbol,
                    "monthly_prices": monthly_dict
                }
        except Exception as e:
            print(f"[ERROR] yahooquery monthly failed: {e}")

        return {"error": f"Could not retrieve monthly data for {symbol}"}


    # def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
    #     try:
    #         print(f"Searching for symbols matching: {symbol_substr}")
    #         url = (
    #             f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol_substr}"
    #             f"&quotes_count=10&news_count=0&lang=en-US&region=US&corsDomain=finance.yahoo.com"
    #         )
    #         headers = {
    #             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    #         }
    #         resp = self.session.get(url, headers=headers)
    #
    #         if resp.status_code != 200:
    #             return [{"error": f"Yahoo search failed with status code {resp.status_code}"}]
    #
    #         results = resp.json()
    #         if "quotes" not in results:
    #             return [{"error": "No matching stocks found"}]
    #
    #         symbols = [q.get("symbol") for q in results["quotes"] if q.get("symbol")]
    #         tickers = Ticker(symbols, session=self.session)
    #         prices = tickers.price
    #
    #         return [
    #             {
    #                 "provider": "yahooquery",
    #                 "company_name": q.get("shortname", "N/A"),
    #                 "symbol": q.get("symbol", "N/A"),
    #                 "latest_price": prices.get(q["symbol"], {}).get("regularMarketPrice", "N/A")
    #             }
    #             for q in results["quotes"]
    #         ]
    #
    #     except Exception as e:
    #         print(f"[ERROR] Exception in get_stocks_data_for_symbol_substring: {e}")
    #         return [{"error": f"Failed to retrieve symbols for '{symbol_substr}'"}]
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

            # First try with yahooquery
            try:
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
            except Exception as yq_error:
                print(f"[WARN] yahooquery failed, falling back to yfinance: {yq_error}")
                # Fallback to yfinance
                data = yf.download(symbols, period="1d", group_by="ticker", threads=False)
                result = []
                for q in results["quotes"]:
                    symbol = q.get("symbol", "N/A")
                    price = (
                        data[symbol]["Close"].iloc[-1]
                        if symbol in data and not data[symbol].empty
                        else "N/A"
                    )
                    result.append({
                        "provider": "yfinance",
                        "company_name": q.get("shortname", "N/A"),
                        "symbol": symbol,
                        "latest_price": price
                    })
                return result

        except Exception as e:
            print(f"[ERROR] Exception in get_stocks_data_for_symbol_substring: {e}")
            return [{"error": f"Failed to retrieve symbols for '{symbol_substr}'"}]

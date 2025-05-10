from datetime import datetime, timedelta
from typing import List, Dict

import yfinance as yahoo_finance_api
from abc import ABC, abstractmethod
from yahooquery import search, Ticker
import requests
from curl_cffi import requests

ALPHA_VANTAGE_KEY = "I0OY1MTU01Z74V0E"




class StockDataProvider(ABC):
    @abstractmethod
    def get_stock_data(self, symbol: str):
        pass

    @abstractmethod
    def get_stocks_data_for_symbol_substring(self, symbol_substr: str):
        pass

# class YahooFinanceProvider(StockDataProvider):
#     """
#     Fetch stock data from Yahoo Finance.
#     """
#     def __init__(self):
#         self.session = requests.Session(impersonate="chrome")
#
#     def get_stock_data(self, symbol: str) -> dict:
#         try:
#             data = yahoo_finance_api.download(symbol, period="1d", progress=False)
#             if data.empty:
#                 return {"error": "Invalid ticker or no data available"}
#
#             stock = yahoo_finance_api.Ticker(symbol)
#             company_name = stock.info.get("longName", "N/A")
#
#             return {
#                 "provider": "Yahoo Finance",
#                 "company_name": company_name,
#                 "symbol": symbol,
#                 "latest_price": data["Close"].iloc[-1][symbol]
#             }
#
#         except Exception as e:
#             return {"error": str(e)}
#
#     def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
#         """
#         Fetches a list of stock symbols matching the given substring from Yahoo Finance,
#         along with the latest recorded price.
#         """
#         try:
#             # Use yahooquery's search function to find matching stocks
#             results = search(symbol_substr)
#
#             if not results or "quotes" not in results:
#                 return [{"error": "No matching stocks found"}]
#
#             result_list = []
#             symbols = [stock.get("symbol", "N/A") for stock in results["quotes"] if stock.get("symbol")]
#
#             # Fetch stock prices for all matched symbols in a single request
#             tickers = Ticker(symbols)
#             prices = tickers.price
#
#             for stock in results["quotes"]:
#                 symbol = stock.get("symbol", "N/A")
#                 company_name = stock.get("shortname", "N/A")
#
#                 # Get latest price from the `price` dictionary
#                 latest_price = prices.get(symbol, {}).get("regularMarketPrice", "N/A")
#
#                 result_list.append({
#                     "provider": "Yahoo Finance",
#                     "company_name": company_name,
#                     "symbol": symbol,
#                     "latest_price": latest_price
#                 })
#
#             return result_list
#
#         except Exception as e:
#             return [{"error": str(e)}]
#
#         except Exception as e:
#             return {"error": str(e)}
#
#     import yfinance as yf
#     import pandas as pd
#     from datetime import datetime
#
#     def get_monthly_close_prices(self, symbol: str):
#         """
#         Fetches the opening and closing stock prices for each month over the last 5 years.
#
#         :param symbol: Stock ticker symbol
#         :return: Dictionary with monthly open and close prices
#         """
#         try:
#             # Fetch stock data for the last 5 years
#             ticker = yahoo_finance_api.Ticker(symbol)
#             data = ticker.history(period="5y", interval="1mo")  # Monthly data
#
#             # Check if DataFrame is empty
#             if data.empty:
#                 return {"error": "No historical data available"}
#
#             # Reset index to get the Date column
#             data.reset_index(inplace=True)
#
#             # Ensure required columns exist
#             required_columns = {"Close"}
#             missing_columns = required_columns - set(data.columns)
#             if missing_columns:
#                 return {"error": f"Missing columns in data: {missing_columns}"}
#
#             # Extract Year-Month format
#             data["Year-Month"] = data["Date"].dt.strftime("%Y-%m")
#
#             # Group by Year-Month: first (Open) and last (Close) price
#             monthly_data = (
#                 data.groupby("Year-Month")
#                 .agg({"Close": "last"})
#                 .to_dict("index")
#             )
#
#             return {
#                 "provider": "yahoo",
#                 "symbol": symbol,
#                 "monthly_prices": monthly_data,
#             }
#
#         except Exception as e:
#             return {"error": str(e)}
class YahooFinanceProvider(StockDataProvider):
    """
    Fetch stock data from Yahoo Finance.
    """

    def __init__(self):
        # Initialize a session from curl_cffi
        self.session = requests.Session(impersonate="chrome")

    def get_stock_data(self, symbol: str) -> dict:
        try:
            # Use yfinance with custom session
            ticker = yahoo_finance_api.Ticker(symbol, session=self.session)
            data = ticker.history(period="1d")

            if data.empty:
                return {"error": "Invalid ticker or no data available"}

            company_name = ticker.info.get("longName", "N/A")

            return {
                "provider": "Yahoo Finance",
                "company_name": company_name,
                "symbol": symbol,
                "latest_price": data["Close"].iloc[-1]  # Corrected here
            }

        except Exception as e:
            return {"error": str(e)}

    import json

    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol_substr}&quotes_count=10&news_count=0&lang=en-US&region=US&corsDomain=finance.yahoo.com"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            resp = self.session.get(url, headers=headers)
            if resp.status_code != 200:
                return [{"error": f"Yahoo search failed with status code {resp.status_code}"}]

            results = resp.json()

            if "quotes" not in results:
                return [{"error": "No matching stocks found"}]

            result_list = []
            symbols = [stock.get("symbol", "N/A") for stock in results["quotes"] if stock.get("symbol")]

            tickers = Ticker(symbols, session=self.session)
            prices = tickers.price

            for stock in results["quotes"]:
                symbol = stock.get("symbol", "N/A")
                company_name = stock.get("shortname", "N/A")
                latest_price = prices.get(symbol, {}).get("regularMarketPrice", "N/A")

                result_list.append({
                    "provider": "Yahoo Finance",
                    "company_name": company_name,
                    "symbol": symbol,
                    "latest_price": latest_price
                })

            return result_list

        except Exception as e:
            return [{"error": str(e)}]

    def get_monthly_close_prices(self, symbol: str):
        """
        Fetches the opening and closing stock prices for each month over the last 5 years.

        :param symbol: Stock ticker symbol
        :return: Dictionary with monthly open and close prices
        """
        try:
            # Use yfinance with custom session to get stock data
            ticker = yahoo_finance_api.Ticker(symbol, session=self.session)
            data = ticker.history(period="5y", interval="1mo")  # Monthly data

            # Check if DataFrame is empty
            if data.empty:
                return {"error": "No historical data available"}

            # Reset index to get the Date column
            data.reset_index(inplace=True)

            # Ensure required columns exist
            required_columns = {"Close"}
            missing_columns = required_columns - set(data.columns)
            if missing_columns:
                return {"error": f"Missing columns in data: {missing_columns}"}

            # Extract Year-Month format
            data["Year-Month"] = data["Date"].dt.strftime("%Y-%m")

            # Group by Year-Month: first (Open) and last (Close) price
            monthly_data = (
                data.groupby("Year-Month")
                .agg({"Close": "last"})
                .to_dict("index")
            )

            return {
                "provider": "yahoo",
                "symbol": symbol,
                "monthly_prices": monthly_data,
            }

        except Exception as e:
            return {"error": str(e)}

class AlphaVantageProvider(StockDataProvider):
    """
    Fetch stock data from Alpha Vantage.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def get_stock_data(self, symbol: str) -> dict:
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "5min",
                "apikey": ALPHA_VANTAGE_KEY,
                "outputsize": "compact",
                "datatype": "json"
            }

            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            if "Time Series (5min)" not in data:
                return {"error": "Invalid ticker or API limit reached"}

            latest_time = max(data["Time Series (5min)"].keys())
            latest_data = data["Time Series (5min)"][latest_time]

            stock = yahoo_finance_api.Ticker(symbol)
            company_name = stock.info.get("longName", "N/A")

            return {
                "provider": "Alpha Vantage",
                "symbol": data["Meta Data"]["2. Symbol"],
                "company_name": company_name,
                "latest_price": float(latest_data["4. close"]),
                # "last_refreshed": data["Meta Data"]["3. Last Refreshed"],
                # "open_price": float(latest_data["1. open"]),
                # "high": float(latest_data["2. high"]),
                # "low": float(latest_data["3. low"]),
                # "volume": int(latest_data["5. volume"]),
                # "interval": data["Meta Data"]["4. Interval"],
                # "time_zone": data["Meta Data"]["6. Time Zone"]
            }

        except Exception as e:
            return {"error": str(e)}

    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        """
        Fetches a list of stock symbols matching the given substring from Alpha Vantage.
        """
        try:
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": symbol_substr,
                "apikey": ALPHA_VANTAGE_KEY
            }

            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            if "bestMatches" not in data:
                return {"error": "No matching stocks found or API limit reached"}

            result = []
            for stock in data["bestMatches"]:
                symbol = stock.get("1. symbol", "N/A")
                company_name = stock.get("2. name", "N/A")
                latest_price = self.get_stock_data(symbol).get("latest_price", 0)

                result.append({
                    "provider": "Alpha Vantage",
                    "company_name": company_name,
                    "symbol": symbol,
                    "latest_price": latest_price
                })

            return result

        except Exception as e:
            return {"error": str(e)}

    def get_monthly_open_close_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Fetches the opening and closing stock prices for each month for the past 5 years.

        :param symbol: Stock ticker symbol
        :return: Dictionary with monthly open and close prices or error message
        """
        try:
            params = {
                "function": "TIME_SERIES_MONTHLY_ADJUSTED",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_KEY,
                "datatype": "json"
            }

            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            # Validate response
            if "Monthly Adjusted Time Series" not in data:
                return {"error": "Invalid ticker or API limit reached"}

            monthly_series = data["Monthly Adjusted Time Series"]
            current_year = datetime.today().year

            # Filter last 5 years of data
            filtered_data = {
                date: {
                    "Close": float(values["4. close"])
                }
                for date, values in monthly_series.items()
                if int(date[:4]) >= current_year - 5  # Filter for last 5 years
            }

            return {
                "provider": "alpha",
                "symbol": symbol,
                "monthly_prices": filtered_data
            }

        except Exception as e:
            return {"error": str(e)}


def get_provider(provider: str):
    providers = {
        "yahoo": YahooFinanceProvider(),
        "alpha": AlphaVantageProvider()
    }

    return providers.get(provider.lower(), None)
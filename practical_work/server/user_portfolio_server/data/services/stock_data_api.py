from datetime import datetime, timedelta
from typing import List, Dict

import yfinance as yahoo_finance_api
from abc import ABC, abstractmethod
from yahooquery import search, Ticker
import requests

ALPHA_VANTAGE_KEY = "I0OY1MTU01Z74V0E"

class StockDataProvider(ABC):
    @abstractmethod
    def get_stock_data(self, symbol: str):
        pass

    @abstractmethod
    def get_stocks_data_for_symbol_substring(self, symbol_substr: str):
        pass

class YahooFinanceProvider(StockDataProvider):
    """
    Fetch stock data from Yahoo Finance.
    """

    def get_stock_data(self, symbol: str) -> dict:
        try:
            data = yahoo_finance_api.download(symbol, period="1d", progress=False)
            if data.empty:
                return {"error": "Invalid ticker or no data available"}

            stock = yahoo_finance_api.Ticker(symbol)
            company_name = stock.info.get("longName", "N/A")

            return {
                "provider": "Yahoo Finance",
                "company_name": company_name,
                "symbol": symbol,
                "latest_price": data["Close"].iloc[-1][symbol]
            }

        except Exception as e:
            return {"error": str(e)}

    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        """
        Fetches a list of stock symbols matching the given substring from Yahoo Finance,
        along with the latest recorded price.
        """
        try:
            # Use yahooquery's search function to find matching stocks
            results = search(symbol_substr)

            if not results or "quotes" not in results:
                return [{"error": "No matching stocks found"}]

            result_list = []
            symbols = [stock.get("symbol", "N/A") for stock in results["quotes"] if stock.get("symbol")]

            # Fetch stock prices for all matched symbols in a single request
            tickers = Ticker(symbols)
            prices = tickers.price

            for stock in results["quotes"]:
                symbol = stock.get("symbol", "N/A")
                company_name = stock.get("shortname", "N/A")

                # Get latest price from the `price` dictionary
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

        except Exception as e:
            return {"error": str(e)}

    def get_monthly_open_close_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Fetches the opening and closing stock prices for each month for the past 5 years.

        :param symbol: Stock ticker symbol
        :return: Dictionary with monthly open and close prices
        """
        try:
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            # Fetch prices for last 5 years
            data = yahoo_finance_api.download(symbol, start=start_date, end=end_date, interval="1mo", progress=False)

            if data.empty:
                return {"error": "No historical data available"}

            # Reset index to get the date column
            data.reset_index(inplace=True)

            # Extract year and month
            data["Year-Month"] = data["Date"].dt.strftime("%Y-%m")

            # Group by year-month and extract first (open) and last (close) price
            monthly_data = data.groupby("Year-Month").agg({"Open": "first", "Close": "last"}).to_dict("index")

            return {
                "provider": "Yahoo Finance",
                "symbol": symbol,
                "monthly_prices": monthly_data
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


def get_provider(provider: str):
    providers = {
        "yahoo": YahooFinanceProvider(),
        "alpha": AlphaVantageProvider()
    }

    return providers.get(provider.lower(), None)
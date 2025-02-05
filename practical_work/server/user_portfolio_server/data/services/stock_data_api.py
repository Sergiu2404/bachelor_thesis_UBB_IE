import yfinance as yahoo_finance_api
from abc import ABC, abstractmethod
import requests

ALPHA_VANTAGE_KEY = "I0OY1MTU01Z74V0E"

class StockDataProvider(ABC):
    @abstractmethod
    def get_stock_data(self, symbol: str):
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


def get_provider(provider: str):
    providers = {
        "yahoo": YahooFinanceProvider(),
        "alpha": AlphaVantageProvider()
    }

    return providers.get(provider.lower(), None)
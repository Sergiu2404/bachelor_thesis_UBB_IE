from data.services.stock_data.stock_provider_interface import StockDataProvider
from datetime import datetime
from typing import List, Dict
import requests
import yfinance as yahoo_finance_api
from data.services.stock_data.stock_provider_interface import StockDataProvider

ALPHA_VANTAGE_KEY = "your_api_key_here"  # Ideally load from env

class AlphaVantageProvider(StockDataProvider):
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

    # def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
    #     """
    #     Fetches a list of stock symbols matching the given substring from Alpha Vantage.
    #     """
    #     try:
    #         params = {
    #             "function": "SYMBOL_SEARCH",
    #             "keywords": symbol_substr,
    #             "apikey": ALPHA_VANTAGE_KEY
    #         }
    #
    #         response = requests.get(self.BASE_URL, params=params)
    #         data = response.json()
    #
    #         if "bestMatches" not in data:
    #             return {"error": "No matching stocks found or API limit reached"}
    #
    #         result = []
    #         for stock in data["bestMatches"]:
    #             symbol = stock.get("1. symbol", "N/A")
    #             company_name = stock.get("2. name", "N/A")
    #             latest_price = self.get_stock_data(symbol).get("latest_price", 0)
    #
    #             result.append({
    #                 "provider": "Alpha Vantage",
    #                 "company_name": company_name,
    #                 "symbol": symbol,
    #                 "latest_price": latest_price
    #             })
    #
    #         return result
    #
    #     except Exception as e:
    #         return {"error": str(e)}
    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        try:
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": symbol_substr,
                "apikey": ALPHA_VANTAGE_KEY
            }

            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            if "bestMatches" not in data:
                return [{"error": "No matching stocks found or API limit reached"}]

            result = []

            # Only fetch price for top 1-2 symbols to avoid hitting rate limits
            for i, stock in enumerate(data["bestMatches"][:2]):
                symbol = stock.get("1. symbol", "N/A")
                company_name = stock.get("2. name", "N/A")
                latest_price = 0.0

                # Fetch daily price only for first 2 to avoid rate limiting
                try:
                    price_params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": symbol,
                        "apikey": ALPHA_VANTAGE_KEY
                    }
                    price_response = requests.get(self.BASE_URL, params=price_params)
                    price_data = price_response.json()

                    if "Time Series (Daily)" in price_data:
                        latest_date = max(price_data["Time Series (Daily)"].keys())
                        latest_price = float(price_data["Time Series (Daily)"][latest_date]["4. close"])
                except Exception as price_error:
                    print(f"[WARN] Failed to fetch price for {symbol}: {price_error}")

                result.append({
                    "provider": "Alpha Vantage",
                    "company_name": company_name,
                    "symbol": symbol,
                    "latest_price": latest_price
                })

            return result

        except Exception as e:
            print(f"[ERROR] AlphaVantageProvider error: {e}")
            return [{"error": str(e)}]

    def get_monthly_close_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
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

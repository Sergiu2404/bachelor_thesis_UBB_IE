from datetime import datetime
from typing import List
from data.services.stock_data.stock_provider_interface import StockDataProvider
import yfinance as yahoo_finance_api
from yahooquery import Ticker
from curl_cffi import requests

class YahooFinanceProvider(StockDataProvider):
    def __init__(self):
        # Initialize a session from curl_cffi
        self.session = requests.Session(impersonate="chrome")

    def get_stock_data(self, symbol: str) -> dict:
        try:
            ticker = yahoo_finance_api.Ticker(symbol, session=self.session)
            data = ticker.history(period="1d")

            if data.empty:
                return {"error": "Invalid ticker or no data available"}

            company_name = ticker.info.get("longName", "N/A")

            return {
                "provider": "Yahoo Finance",
                "company_name": company_name,
                "symbol": symbol,
                "latest_price": data["Close"].iloc[-1]
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

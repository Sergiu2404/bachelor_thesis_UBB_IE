from data.services.stock_data.stock_provider_interface import StockDataProvider
from typing import List, Dict
import requests
from datetime import datetime

FMP_API_KEY = "EWuWTAQL1XH0h9olxaxbtmDyUSKfTmYd"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

class FinancialModelingPrepProvider(StockDataProvider):

    def get_stock_data(self, symbol: str) -> dict:
        try:
            url = f"{FMP_BASE_URL}/quote/{symbol}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            data = response.json()

            if not data or "price" not in data[0]:
                return {"error": f"No data found for {symbol}"}

            return {
                "provider": "FMP",
                "symbol": symbol,
                "company_name": data[0].get("name", "N/A"),
                "latest_price": data[0]["price"]
            }
        except Exception as e:
            return {"error": str(e)}

    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        try:
            url = f"{FMP_BASE_URL}/search?query={symbol_substr}&limit=5&exchange=NASDAQ&apikey={FMP_API_KEY}"
            response = requests.get(url)
            matches = response.json()

            result = []
            for match in matches:
                symbol = match.get("symbol")
                name = match.get("name", "N/A")
                stock_data = self.get_stock_data(symbol)

                result.append({
                    "provider": "FMP",
                    "company_name": name,
                    "symbol": symbol,
                    "latest_price": stock_data.get("latest_price", 0)
                })

            return result
        except Exception as e:
            return [{"error": str(e)}]

    def get_monthly_close_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
        try:
            url = f"{FMP_BASE_URL}/historical-price-full/{symbol}?serietype=line&apikey={FMP_API_KEY}"
            response = requests.get(url)
            data = response.json()

            if "historical" not in data:
                return {"error": "No historical data available"}

            historical = data["historical"]
            monthly_data = {}

            for entry in historical:
                date = entry["date"]
                year_month = date[:7]
                price = entry["close"]

                if year_month not in monthly_data:
                    monthly_data[year_month] = {"Close": price}

            five_years_ago = datetime.today().year - 5
            filtered_data = {
                ym: data
                for ym, data in monthly_data.items()
                if int(ym.split("-")[0]) >= five_years_ago
            }

            return {
                "provider": "FMP",
                "symbol": symbol,
                "monthly_prices": filtered_data
            }

        except Exception as e:
            return {"error": str(e)}

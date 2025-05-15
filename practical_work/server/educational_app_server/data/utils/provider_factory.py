from data.services.stock_data.yahoo_finance_provider import YahooFinanceProvider
from data.services.stock_data.alpha_vantage_provider import AlphaVantageProvider
from data.services.stock_data.stock_provider_interface import StockDataProvider
from typing import Optional, Union

def get_provider(provider: str) -> Optional[StockDataProvider]:
    providers = {
        "yahoo": YahooFinanceProvider(),
        "alpha": AlphaVantageProvider()
    }
    return providers.get(provider.lower())

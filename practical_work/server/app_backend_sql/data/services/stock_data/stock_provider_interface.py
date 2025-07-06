from abc import ABC, abstractmethod
from typing import List, Dict

class StockDataProvider(ABC):
    @abstractmethod
    def get_stock_data(self, symbol: str) -> dict:
        pass

    @abstractmethod
    def get_stocks_data_for_symbol_substring(self, symbol_substr: str) -> List[dict]:
        pass

    @abstractmethod
    def get_monthly_close_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
        pass

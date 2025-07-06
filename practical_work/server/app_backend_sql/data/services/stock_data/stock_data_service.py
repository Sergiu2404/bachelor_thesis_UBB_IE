from data.services.stock_data.stock_provider_interface import StockDataProvider

class StockDataService:
    def __init__(self, provider: StockDataProvider):
        self.provider = provider

    def get_stock_data(self, symbol: str):
        return self.provider.get_stock_data(symbol)

    def get_matching_stocks(self, symbol_substr: str):
        return self.provider.get_stocks_data_for_symbol_substring(symbol_substr)

    def get_monthly_prices(self, symbol: str):
        return self.provider.get_monthly_close_prices(symbol)

from fredapi import Fred

FRED_API_KEY = "15a4d2e1121e1a09cc3021690a867b13"

class FredAPI:
    def __init__(self):
        self.fred = Fred(FRED_API_KEY)
    def get_gdp_history(self):
        return self.fred.get_series("GDP")
    def get_consumer_price_index_history(self):
        return self.fred.get_series("CPIAUCSL")

    def get_unemployment_rate_history(self):
        return self.fred.get_series("UNRATE")

    def get_annual_inflation_rate_consumer_price_history(self):
        return self.fred.get_series("FPCPITOTLZGUSA")
        #return self.fred.get_series("T10YIE")

    def get_market_expectation_inflation_rate(self):
        return self.fred.get_series("T10YIE")

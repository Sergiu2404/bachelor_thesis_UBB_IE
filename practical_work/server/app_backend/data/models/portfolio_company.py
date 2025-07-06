from bson import ObjectId
from pydantic import BaseModel

class PortfolioCompany(BaseModel):
    username: str
    symbol: str
    company_name: str
    quantity: int
    average_buy_price: float
    total_current_value: float

    @staticmethod
    def serialize_portfolio_company(company_data):
        if isinstance(company_data["_id"], ObjectId):
            company_data["_id"] = str(company_data["_id"])
        return company_data
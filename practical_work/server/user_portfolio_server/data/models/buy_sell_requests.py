from pydantic import BaseModel


class BuySellStockRequest(BaseModel):
    symbol: str
    quantity: int


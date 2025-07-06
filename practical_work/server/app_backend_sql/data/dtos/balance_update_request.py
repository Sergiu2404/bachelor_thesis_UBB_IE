from pydantic import BaseModel

class BalanceUpdateRequest(BaseModel):
    new_balance: float
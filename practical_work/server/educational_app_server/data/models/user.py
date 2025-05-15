from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    username: Optional[str] = None
    expires_in: Optional[int] = None

class User(BaseModel):
    username: str
    email: EmailStr
    virtual_money_balance: float = Field(default=15000.00, ge=0)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserInDB(User):
    hashed_password: str
from datetime import datetime, timedelta
from typing import Optional, Set

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

from data.models.user import User
from data.dtos.user_dtos import UserRegister, UserLogin
from data.database.database import mongodb
from data.dtos.balance_update_request import BalanceUpdateRequest

SECRET_KEY = "7317758526d8838bc182a5226a6172153146c2d2e4ddf7e9a4e71627783e48e3"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 180
BLACKLIST: Set[str] = set()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationService:

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    async def get_user_by_username(username: str) -> Optional[User]:
        user_data = await mongodb.users.find_one({"username": username})
        return User(**user_data) if user_data else None

    @staticmethod
    async def register_user(user: UserRegister) -> User:
        if await mongodb.users.find_one({"username": user.username}):
            raise HTTPException(status_code=400, detail="Username already registered")

        if await mongodb.users.find_one({"email": user.email}):
            raise HTTPException(status_code=400, detail="Email already registered")

        user_dict = user.dict()
        user_dict["hashed_password"] = AuthenticationService.hash_password(user_dict.pop("password"))
        user_dict["virtual_money_balance"] = 15000.00

        new_user = await mongodb.users.insert_one(user_dict)
        created_user = await mongodb.users.find_one({"_id": new_user.inserted_id})
        return User(**created_user)

    @staticmethod
    async def login_user(user: UserLogin):
        db_user = await mongodb.users.find_one({"username": user.username})
        if not db_user or not AuthenticationService.verify_password(user.password, db_user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        token = AuthenticationService.create_access_token(data={"sub": user.username})
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": user.username,
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }

    @staticmethod
    def is_token_blacklisted(token: str) -> bool:
        return token in BLACKLIST

    @staticmethod
    def blacklist_token(token: str):
        BLACKLIST.add(token)

    @staticmethod
    async def update_balance(current_user: User, balance_update: BalanceUpdateRequest):
        if balance_update.new_balance < 0:
            raise HTTPException(status_code=400, detail="Balance cannot be negative")

        result = await mongodb.users.update_one(
            {"username": current_user.username},
            {"$set": {"virtual_money_balance": balance_update.new_balance}}
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Failed to update balance")

        return {"message": "Balance updated successfully", "new_balance": balance_update.new_balance}

from datetime import datetime, timedelta
from typing import Optional, Set

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from data.models.models_sql_alchemy import User as DBUser  # SQLAlchemy model
from data.database.db import get_db  # get_db() from db.py
from data.dtos.user_dtos import UserRegister, UserLogin
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
    async def get_user_by_username(username: str, db: AsyncSession) -> Optional[DBUser]:
        result = await db.execute(select(DBUser).where(DBUser.username == username))
        return result.scalars().first()

    @staticmethod
    async def register_user(user: UserRegister, db: AsyncSession) -> DBUser:
        result = await db.execute(select(DBUser).where(DBUser.username == user.username))
        if result.scalars().first():
            raise HTTPException(status_code=400, detail="Username already registered")

        result = await db.execute(select(DBUser).where(DBUser.email == user.email))
        if result.scalars().first():
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_pw = AuthenticationService.hash_password(user.password)
        new_user = DBUser(
            username=user.username,
            email=user.email,
            hashed_password=hashed_pw,
            virtual_money_balance=15000.00
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user

    @staticmethod
    async def login_user(user: UserLogin, db: AsyncSession):
        result = await db.execute(select(DBUser).where(DBUser.username == user.username))
        db_user = result.scalars().first()

        if not db_user or not AuthenticationService.verify_password(user.password, db_user.hashed_password):
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
    async def update_balance(current_user: DBUser, balance_update: BalanceUpdateRequest, db: AsyncSession):
        if balance_update.new_balance < 0:
            raise HTTPException(status_code=400, detail="Balance cannot be negative")

        current_user.virtual_money_balance = balance_update.new_balance
        db.add(current_user)
        await db.commit()
        return {"message": "Balance updated successfully", "new_balance": balance_update.new_balance}

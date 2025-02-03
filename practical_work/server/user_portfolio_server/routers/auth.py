from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from data.db.database import users_collection

# JWT & Security Configuration
SECRET_KEY = "7317758526d8838bc182a5226a6172153146c2d2e4ddf7e9a4e71627783e48e3"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# FastAPI Router
router = APIRouter(prefix="/auth", tags=["auth"])

# User Models
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: EmailStr
    virtual_money_balance: float

# Utility Functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Authentication Endpoints
@router.post("/register", response_model=User)
async def register(user: UserRegister):
    existing_user = await users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    existing_email = await users_collection.find_one({"email": user.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.password)
    virtual_money_balance = 20000.00  # Default balance for new users

    new_user = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "virtual_money_balance": virtual_money_balance
    }

    await users_collection.insert_one(new_user)

    return {"username": user.username, "email": user.email, "virtual_money_balance": virtual_money_balance}

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    db_user = await users_collection.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/profile", response_model=User)
async def get_profile(username: str):
    user = await users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"username": user["username"], "email": user["email"], "virtual_money_balance": user["virtual_money_balance"]}

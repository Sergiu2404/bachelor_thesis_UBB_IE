from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt

from data.models.balance_update_request import BalanceUpdateRequest
from data.models.user import UserRegister, UserLogin, Token, User, UserInDB
from data.db.database import users_collection
from fastapi.security import OAuth2PasswordRequestForm


SECRET_KEY = "7317758526d8838bc182a5226a6172153146c2d2e4ddf7e9a4e71627783e48e3"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 180

BLACKLIST = set()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

router = APIRouter(prefix="/auth", tags=["auth"])


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
#
#     user = await users_collection.find_one({"username": username})
#     if user is None:
#         raise credentials_exception
#     return UserInDB(**user)



# async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
#
#     user = await get_user_by_username(username)
#     if user is None:
#         raise credentials_exception
#     return user





# update the login endpoint to handle form data as well as JSON
@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    db_user = await users_collection.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # create token with extended expiration
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    # return token with additional user info
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    }


# TODO test if function below works with the BLACKLIST
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    if token in BLACKLIST:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been blacklisted, please log in again",
            headers={"WWW-Authenticate": "Bearer"},
        )
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user), token: str = Depends(oauth2_scheme)):
    """
    Logout the current user and blacklist the token.
    """
    BLACKLIST.add(token)  # Add token to blacklist

    return {"message": "Logged out successfully"}






# update register endpoint to return more information
@router.post("/register", response_model=User)
async def register(user: UserRegister):
    # check if username exists
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    user_dict = user.dict()
    user_dict["hashed_password"] = hash_password(user_dict.pop("password"))
    user_dict["virtual_money_balance"] = 15000.00

    try:
        new_user = await users_collection.insert_one(user_dict)
        created_user = await users_collection.find_one({"_id": new_user.inserted_id})
        return User(**created_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

async def get_user_by_username(username: str) -> Optional[User]:
    user_data = await users_collection.find_one({"username": username})

    print(user_data)
    if user_data:
        return User(**user_data)
    return None

# check if authentication made
# @router.get("/test-auth")
# async def test_auth(current_user: UserInDB = Depends(get_current_user)):
#     return {"message": "Authentication successful", "username": current_user.username}
@router.get("/connected-user")
async def test_auth(current_user: User = Depends(get_current_user)):
    return User(
        username=current_user.username,
        email=current_user.email,
        virtual_money_balance=current_user.virtual_money_balance,
        created_at=current_user.created_at
    )

@router.post("/update-balance")
async def update_balance(balance_update: BalanceUpdateRequest, current_user: User = Depends(get_current_user)):
    if balance_update.new_balance < 0:
        raise HTTPException(status_code=400, detail="Balance cannot be negative")

    query_filter = {"username": current_user.username}
    update_operation = {
        "$set": {
            "virtual_money_balance": balance_update.new_balance
        }
    }
    result = await users_collection.update_one(
        query_filter,
        update_operation
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Failed to update balance of the user")

    return {"message": "Balance updated succesfully", "new_balance": balance_update.new_balance}
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional

from data.models.user import User, Token
from data.dtos.user_dtos import UserRegister, UserLogin
from data.dtos.balance_update_request import BalanceUpdateRequest
from data.services.authentication.authentication_service import AuthenticationService, SECRET_KEY, ALGORITHM

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

authentication_router = APIRouter(prefix="/auth", tags=["auth"])

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    if AuthenticationService.is_token_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token has been blacklisted")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await AuthenticationService.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@authentication_router.post("/register", response_model=User)
async def register(user: UserRegister):
    return await AuthenticationService.register_user(user)


@authentication_router.post("/login", response_model=Token)
async def login(user: UserLogin):
    return await AuthenticationService.login_user(user)


@authentication_router.post("/logout")
async def logout(current_user: User = Depends(get_current_user), token: str = Depends(oauth2_scheme)):
    AuthenticationService.blacklist_token(token)
    return {"message": "Logged out successfully"}


@authentication_router.get("/connected-user", response_model=User)
async def get_connected_user(current_user: User = Depends(get_current_user)):
    return current_user


@authentication_router.post("/update-balance")
async def update_balance(
    balance_update: BalanceUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    return await AuthenticationService.update_balance(current_user, balance_update)

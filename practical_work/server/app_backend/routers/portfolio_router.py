from typing import List
from fastapi import APIRouter, Depends, HTTPException

from data.dtos.buy_sell_requests import BuySellStockRequest
from data.models.user import User
from data.services.portfolio.portfolio_service import PortfolioService
from routers.authentication_router import get_current_user

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@portfolio_router.get("/{username}", response_model=List[dict])
async def get_portfolio_for_user(username: str, current_user: User = Depends(get_current_user)):
    #return await PortfolioService.get_user_portfolio(username)
    if username != current_user.username:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await PortfolioService.get_user_portfolio(username)


@portfolio_router.post("/{username}/buy")
async def buy_stock(username: str, request: BuySellStockRequest, current_user: User = Depends(get_current_user)):
    #return await PortfolioService.buy_stock(username, request)
    print(username, current_user.username)
    if username != current_user.username:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await PortfolioService.buy_stock(username, request)


@portfolio_router.post("/{username}/sell")
async def sell_stock(username: str, request: BuySellStockRequest, current_user: User = Depends(get_current_user)):
    print(type(username), type(request.symbol), type(request.quantity))
    #return await PortfolioService.sell_stock(username, request)
    if username != current_user.username:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await PortfolioService.sell_stock(username, request)

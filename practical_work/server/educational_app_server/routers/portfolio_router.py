from typing import List
from fastapi import APIRouter

from data.dtos.buy_sell_requests import BuySellStockRequest
from data.services.portfolio.portfolio_service import PortfolioService

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@portfolio_router.get("/{username}", response_model=List[dict])
async def get_portfolio_for_user(username: str):
    return await PortfolioService.get_user_portfolio(username)


@portfolio_router.post("/{username}/buy")
async def buy_stock(username: str, request: BuySellStockRequest):
    return await PortfolioService.buy_stock(username, request)


@portfolio_router.post("/{username}/sell")
async def sell_stock(username: str, request: BuySellStockRequest):
    print(type(username), type(request.symbol), type(request.quantity))

    return await PortfolioService.sell_stock(username, request)

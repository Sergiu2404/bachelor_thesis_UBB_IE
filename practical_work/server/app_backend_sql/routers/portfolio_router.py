from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from data.dtos.buy_sell_requests import BuySellStockRequest
from data.models.models_sql_alchemy import User as DBUser  # SQLAlchemy model
from data.services.portfolio.portfolio_service import PortfolioService
from routers.authentication_router import get_current_user
from data.database.db import get_db

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@portfolio_router.get("/{username}", response_model=List[dict])
async def get_portfolio_for_user(
    username: str,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if username != current_user.username:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await PortfolioService.get_user_portfolio(username, db)


@portfolio_router.post("/{username}/buy")
async def buy_stock(
    username: str,
    request: BuySellStockRequest,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if username != current_user.username:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await PortfolioService.buy_stock(username, request, db)


@portfolio_router.post("/{username}/sell")
async def sell_stock(
    username: str,
    request: BuySellStockRequest,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if username != current_user.username:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await PortfolioService.sell_stock(username, request, db)

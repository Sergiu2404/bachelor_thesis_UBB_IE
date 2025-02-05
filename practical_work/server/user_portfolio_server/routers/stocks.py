from fastapi import APIRouter, HTTPException
from data.services.stock_data_api import get_provider

stocks_router = APIRouter(prefix="/stocks", tags=["stocks"])

@stocks_router.get("/{provider}/{symbol}")
async def get_stock_data(provider: str, symbol: str):
    stock_provider = get_provider(provider)

    if not stock_provider:
        raise HTTPException(status_code=400, detail="invalid provider, use 'yahoo' or 'alpha'")

    return stock_provider.get_stock_data(symbol)
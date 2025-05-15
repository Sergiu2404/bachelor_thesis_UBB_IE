from fastapi import APIRouter, HTTPException
from data.utils.provider_factory import get_provider
from data.services.stock_data.stock_data_service import StockDataService

stock_data_router = APIRouter(prefix="/stocks", tags=["stocks"])

@stock_data_router.get("/stock/{provider}/{symbol}")
async def get_stock_data(provider: str, symbol: str):
    stock_provider = get_provider(provider)
    if not stock_provider:
        raise HTTPException(status_code=400, detail="Invalid provider, use 'yahoo' or 'alpha'")

    service = StockDataService(stock_provider)
    return service.get_stock_data(symbol)


@stock_data_router.get("/{provider}")
async def get_stocks_data_for_symbol_substring(provider: str, symbol_substr: str):
    stock_provider = get_provider(provider)
    if not stock_provider:
        raise HTTPException(status_code=400, detail="Invalid provider, use 'yahoo' or 'alpha'")

    service = StockDataService(stock_provider)
    return service.get_matching_stocks(symbol_substr)


@stock_data_router.get("/monthly/{provider}/{symbol}")
async def get_monthly_stock_data(provider: str, symbol: str):
    stock_provider = get_provider(provider)
    if not stock_provider:
        raise HTTPException(status_code=400, detail="Invalid provider, only 'yahoo' or 'alpha' providers supported for now")

    service = StockDataService(stock_provider)
    return service.get_monthly_prices(symbol)
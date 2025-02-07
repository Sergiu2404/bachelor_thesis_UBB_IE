from fastapi import APIRouter, HTTPException
from data.services.stock_data_api import get_provider

stocks_router = APIRouter(prefix="/stocks", tags=["stocks"])

@stocks_router.get("/stock/{provider}/{symbol}")
async def get_stock_data(provider: str, symbol: str):
    stock_provider = get_provider(provider)

    if not stock_provider:
        raise HTTPException(status_code=400, detail="invalid provider, use 'yahoo' or 'alpha'")

    return stock_provider.get_stock_data(symbol)

@stocks_router.get("/{provider}")
async def get_stocks_data_for_symbol_substring(provider: str, symbol_substr: str):
    stock_provider = get_provider(provider)

    if not stock_provider:
        raise HTTPException(status_code=400, detail="invalid provider, use 'yahoo' or 'alpha'")

    #print(stock_provider.get_stocks_data_for_symbol_substring(symbol_substr))
    return stock_provider.get_stocks_data_for_symbol_substring(symbol_substr)

@stocks_router.get("/monthly/{provider}/{symbol}")
async def get_monthly_stock_data(provider: str, symbol: str):
    """
    Fetch monthly open and close stock prices for the past 5 years.
    """
    stock_provider = get_provider(provider)

    # if not stock_provider or provider.lower() != "yahoo":
    #    raise HTTPException(status_code=400, detail="Invalid provider, only 'yahoo' is supported for this endpoint")
    if not stock_provider:
        raise HTTPException(status_code=400, detail="Invalid provider, only 'yahoo' or 'alpha' providers supported for now")

    #print(stock_provider.get_monthly_open_close_prices(symbol))
    return stock_provider.get_monthly_close_prices(symbol)

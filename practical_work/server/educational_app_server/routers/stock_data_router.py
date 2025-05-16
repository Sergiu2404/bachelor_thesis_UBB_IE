from fastapi import APIRouter, HTTPException, Depends
from starlette import status

from data.models.user import User
from data.utils.provider_factory import get_provider
from data.services.stock_data.stock_data_service import StockDataService
from routers.authentication_router import get_current_user

stock_data_router = APIRouter(prefix="/stocks", tags=["stocks"])

@stock_data_router.get("/stock/{provider}/{symbol}")
async def get_stock_data(provider: str, symbol: str, current_user: User = Depends(get_current_user)):
    try:
        stock_provider = get_provider(provider)
        if not stock_provider:
            raise HTTPException(status_code=400, detail="Invalid provider, use 'yahoo' or 'alpha'")

        service = StockDataService(stock_provider)
        return service.get_stock_data(symbol)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stock data of {symbol} from {provider}: {str(e)}"
        )


@stock_data_router.get("/{provider}")
async def get_stocks_data_for_symbol_substring(provider: str, symbol_substr: str, current_user: User = Depends(get_current_user)):
    try:
        stock_provider = get_provider(provider)
        if not stock_provider:
            raise HTTPException(status_code=400, detail="Invalid provider, use 'yahoo' or 'alpha'")

        service = StockDataService(stock_provider)
        return service.get_matching_stocks(symbol_substr)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stock data from {provider} for the given substring {symbol_substr}: {str(e)}"
        )

@stock_data_router.get("/monthly/{provider}/{symbol}")
async def get_monthly_stock_data(provider: str, symbol: str, current_user: User = Depends(get_current_user)):
    try:
        stock_provider = get_provider(provider)
        if not stock_provider:
            raise HTTPException(status_code=400, detail="Invalid provider, only 'yahoo' or 'alpha' providers supported for now")

        service = StockDataService(stock_provider)
        return service.get_monthly_prices(symbol)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch monthly stock data about {symbol} from {provider}: {str(e)}"
        )
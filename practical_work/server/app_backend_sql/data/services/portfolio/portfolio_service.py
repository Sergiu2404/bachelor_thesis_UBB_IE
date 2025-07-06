from typing import List, Dict
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from data.models.models_sql_alchemy import User, PortfolioCompany
from data.dtos.buy_sell_requests import BuySellStockRequest
import yfinance as yf
from yahooquery import Ticker


class PortfolioService:

    @staticmethod
    async def get_current_stock_price(symbol: str) -> float:
        try:
            df = yf.download(symbol, period="1d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
            raise Exception("Empty data from yfinance")
        except Exception:
            try:
                ticker = Ticker(symbol)
                info = ticker.price.get(symbol)
                return float(info["regularMarketPrice"])
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to fetch price for {symbol}")

    @staticmethod
    async def get_company_name(symbol: str) -> str:
        try:
            name = yf.Ticker(symbol).info.get("longName", "N/A")
            return name
        except Exception:
            try:
                ticker = Ticker(symbol)
                info = ticker.price.get(symbol)
                return info.get("longName") or info.get("shortName") or "N/A"
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to fetch company name for {symbol}")

    @staticmethod
    async def get_user_portfolio(username: str, db: AsyncSession) -> List[dict]:
        result = await db.execute(select(PortfolioCompany).where(PortfolioCompany.username == username))
        portfolio = result.scalars().all()
        return [
            {
                "username": p.username,
                "symbol": p.symbol,
                "company_name": p.company_name,
                "quantity": p.quantity,
                "average_buy_price": p.average_buy_price,
                "total_current_price": p.total_current_value
            } for p in portfolio
        ]

    @staticmethod
    async def buy_stock(username: str, request: BuySellStockRequest, db: AsyncSession) -> Dict:
        symbol = request.symbol
        quantity = request.quantity

        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be >= 0")

        current_price = await PortfolioService.get_current_stock_price(symbol)

        result = await db.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=400, detail="User not found")

        total_cost = current_price * quantity
        if user.virtual_money_balance < total_cost:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        result = await db.execute(select(PortfolioCompany).where(
            PortfolioCompany.username == username, PortfolioCompany.symbol == symbol))
        existing_stock = result.scalars().first()

        if existing_stock:
            total_quantity = existing_stock.quantity + quantity
            total_spent = (existing_stock.average_buy_price * existing_stock.quantity) + (current_price * quantity)
            new_avg_price = total_spent / total_quantity
            existing_stock.quantity = total_quantity
            existing_stock.average_buy_price = new_avg_price
            existing_stock.total_current_value = total_quantity * current_price
        else:
            company_name = await PortfolioService.get_company_name(symbol)
            new_entry = PortfolioCompany(
                username=username,
                symbol=symbol,
                company_name=company_name,
                quantity=quantity,
                average_buy_price=current_price,
                total_current_value=quantity * current_price
            )
            db.add(new_entry)

        user.virtual_money_balance -= total_cost

        try:
            await db.commit()
        except Exception as e:
            await db.rollback()
            raise HTTPException(status_code=500, detail=f"Database commit failed: {str(e)}")
        # return {
        #     "message": "Stock purchase recorded successfully",
        #     "bought_at_price": current_price,
        #     "total_cost": total_cost
        # }
        return {
            "message": "Stock purchase recorded successfully",
            "username": username,
            "symbol": symbol,
            "quantity": quantity,
            "bought_at_price": current_price,
            "total_cost": total_cost,
            "total_current_price": quantity * current_price  # match frontend expectation
        }

    @staticmethod
    async def sell_stock(username: str, request: BuySellStockRequest, db: AsyncSession) -> Dict:
        symbol = request.symbol
        quantity = request.quantity

        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be > 0")

        result = await db.execute(select(PortfolioCompany).where(
            PortfolioCompany.username == username, PortfolioCompany.symbol == symbol))
        stock = result.scalars().first()

        if not stock:
            raise HTTPException(status_code=400, detail="Stock not found in the portfolio")
        if quantity > stock.quantity:
            raise HTTPException(status_code=400, detail="Cannot sell more than you own")

        current_price = await PortfolioService.get_current_stock_price(symbol)
        total_sale_value = quantity * current_price
        remaining_quantity = stock.quantity - quantity

        result = await db.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=400, detail="User not found")

        user.virtual_money_balance += total_sale_value

        if remaining_quantity == 0:
            await db.delete(stock)
        else:
            stock.quantity = remaining_quantity
            stock.total_current_value = remaining_quantity * current_price

        await db.commit()
        return {
            "message": "Stock sale recorded successfully" if remaining_quantity else "All shares of this stock removed",
            "sale_value": total_sale_value
        }

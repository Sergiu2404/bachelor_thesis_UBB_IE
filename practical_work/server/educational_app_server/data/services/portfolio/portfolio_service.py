from typing import List, Dict

import yfinance as yahoo_finance_api
from fastapi import HTTPException

from data.database.database import mongodb
from data.dtos.buy_sell_requests import BuySellStockRequest
from data.models.portfolio_company import PortfolioCompany


# class PortfolioService:
#     @staticmethod
#     async def get_current_stock_price(symbol: str) -> float:
#         try:
#             stock_data = yahoo_finance_api.Ticker(symbol)
#             print(f"get_current_stock_price: {stock_data.history(period='1d')['Close'].iloc[-1]}")
#             return float(stock_data.history(period='1d')['Close'].iloc[-1])
#         except Exception:
#             raise HTTPException(status_code=500, detail=f"Failed to fetch price for {symbol}")
#
#     @staticmethod
#     async def get_company_name(symbol: str) -> str:
#         try:
#             stock_data = yahoo_finance_api.Ticker(symbol)
#             return stock_data.info.get("longName", "N/A")
#         except Exception:
#             raise HTTPException(status_code=500, detail=f"Failed to fetch company name for {symbol}")


from yahooquery import Ticker
import yfinance as yf
import pandas as pd
from fastapi import HTTPException

class PortfolioService:
    @staticmethod
    async def get_current_stock_price(symbol: str) -> float:
        # yfinance
        try:
            print(f"[INFO] Trying yfinance.download for {symbol}")
            df = yf.download(symbol, period="1d")
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                print(f"[SUCCESS] yfinance price for {symbol}: {price}")
                return price
            else:
                raise Exception("Empty data from yfinance")
        except Exception as e:
            print(f"[WARNING] yfinance failed: {e}")

        # yahooquery
        try:
            print(f"[INFO] Trying yahooquery for {symbol}")
            ticker = Ticker(symbol)
            price_info = ticker.price.get(symbol)
            if price_info and "regularMarketPrice" in price_info:
                price = float(price_info["regularMarketPrice"])
                print(f"[SUCCESS] yahooquery price for {symbol}: {price}")
                return price
            else:
                raise Exception("regularMarketPrice not found")
        except Exception as e:
            print(f"[ERROR] yahooquery failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch price for {symbol}")

    @staticmethod
    async def get_company_name(symbol: str) -> str:
        # 1. Try yfinance
        try:
            print(f"[INFO] Trying yfinance.Ticker.info for {symbol}")
            info = yf.Ticker(symbol).info
            name = info.get("longName", "N/A")
            print(f"[SUCCESS] yfinance company name for {symbol}: {name}")
            return name
        except Exception as e:
            print(f"[WARNING] yfinance failed for company name: {e}")

        # 2. Try yahooquery
        try:
            print(f"[INFO] Trying yahooquery for company name of {symbol}")
            ticker = Ticker(symbol)
            info = ticker.price.get(symbol)
            name = info.get("longName") or info.get("shortName") or "N/A"
            print(f"[SUCCESS] yahooquery company name: {name}")
            return name
        except Exception as e:
            print(f"[ERROR] yahooquery failed for company name: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch company name for {symbol}")

    @staticmethod
    async def get_user_portfolio(username: str) -> List[dict]:
        portfolio_companies = await mongodb.portfolio.find({"username": username}).to_list(None)
        return [PortfolioCompany.serialize_portfolio_company(c) for c in portfolio_companies] if portfolio_companies else []

    @staticmethod
    async def buy_stock(username: str, request: BuySellStockRequest) -> Dict:
        symbol = request.symbol
        quantity = request.quantity

        print(symbol, quantity)

        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be >= 0")

        current_price = await PortfolioService.get_current_stock_price(symbol)
        print("get_current_price done")
        user = await mongodb.users.find_one({"username": username})
        if not user:
            raise HTTPException(status_code=400, detail="User doesn't exist")

        total_cost = current_price * quantity
        if user["virtual_money_balance"] < total_cost:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        existing_stock = await mongodb.portfolio.find_one({"username": username, "symbol": symbol})

        if existing_stock:
            total_quantity = existing_stock["quantity"] + quantity
            total_spent = (existing_stock["average_buy_price"] * existing_stock["quantity"]) + (current_price * quantity)
            new_avg_price = total_spent / total_quantity
            total_current_value = total_quantity * current_price

            await mongodb.portfolio.update_one(
                {"username": username, "symbol": symbol},
                {"$set": {
                    "quantity": total_quantity,
                    "average_buy_price": new_avg_price,
                    "total_current_price": total_current_value
                }}
            )
        else:
            company_name = await PortfolioService.get_company_name(symbol)
            await mongodb.portfolio.insert_one({
                "username": username,
                "symbol": symbol,
                "company_name": company_name,
                "quantity": quantity,
                "average_buy_price": current_price,
                "total_current_price": current_price * quantity
            })

        await mongodb.users.update_one(
            {"username": username},
            {"$inc": {"virtual_money_balance": -total_cost}}
        )

        return {
            "message": "Stock purchase recorded successfully",
            "bought_at_price": current_price,
            "total_cost": total_cost
        }

    @staticmethod
    async def sell_stock(username: str, request: BuySellStockRequest) -> Dict:
        print("trecut de router")
        symbol = request.symbol
        quantity = request.quantity

        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity to be sold must be > 0")

        existing_stock = await mongodb.portfolio.find_one({"username": username, "symbol": symbol})
        print(f"existing stock: {existing_stock}")
        if not existing_stock:
            raise HTTPException(status_code=400, detail="Stock not found in the portfolio")
        if quantity > existing_stock["quantity"]:
            raise HTTPException(status_code=400, detail="Cannot sell more than you own")

        current_price = await PortfolioService.get_current_stock_price(symbol)
        print(f"current price: {current_price}")
        total_sale_value = quantity * current_price
        remaining_quantity = existing_stock["quantity"] - quantity

        print(f"sale value: {total_sale_value},remaining quantity:  {remaining_quantity}")

        await mongodb.users.update_one(
            {"username": username},
            {"$inc": {"virtual_money_balance": total_sale_value}}
        )
        print("users updated")

        if remaining_quantity == 0:
            await mongodb.portfolio.delete_one({"username": username, "symbol": symbol})
            print("portfolio_company deleted (quantity is 0)")
            return {
                "message": "All shares of this stock removed from the portfolio",
                "sale_value": total_sale_value
            }
        else:
            await mongodb.portfolio.update_one(
                {"username": username, "symbol": symbol},
                {"$set": {
                    "quantity": remaining_quantity,
                    "total_current_price": remaining_quantity * current_price
                }}
            )
            print(f"quantity updated for user")
            return {
                "message": "Stock sale recorded successfully",
                "sale_value": total_sale_value
            }

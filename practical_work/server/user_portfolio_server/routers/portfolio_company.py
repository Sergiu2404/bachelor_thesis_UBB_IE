from typing import List

from fastapi import APIRouter, HTTPException
import yfinance as yahoo_finance_api

from data.db.database import portfolio_company_collection
from data.db.database import users_collection

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])

async def get_current_stock_price_company(symbol: str) -> float:
    try:
        stock_data = yahoo_finance_api.Ticker(symbol)
        current_price = stock_data.history(period="1d")["Close"].iloc[-1]
        return float(current_price)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Failed to fetch price for {symbol}")

async def get_company_name_for_symbol(symbol: str) -> str:
    try:
        stock_data = yahoo_finance_api.Ticker(symbol)
        company_name = stock_data.info.get("longName", "N/A")

        return company_name
    except Exception:
        raise HTTPException(status_code=500, detail=f"Failed to fetch company name for {symbol}")


@portfolio_router.get("/{username}", response_model=List[dict])
async def get_portfolio_for_user(username: str):
    portfolio_companies = []
    portfolio_companies = await portfolio_company_collection.find({"username": username}).to_list(length=None)

    if not portfolio_companies:
        return []

    return portfolio_companies

@portfolio_router.post("/{username}/buy")
async def buy_stock(username: str, symbol: str, quantity: int):
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be an integer >= 0")

    current_price = await get_current_stock_price_company(symbol)
    if not current_price:
        raise HTTPException(status_code=400, detail="Could not retrieve current stock price from stock data")

    user = await users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=400, detail="User doesnt exist")

    total_cost = current_price * quantity
    if user["virtual_money_balance"] < total_cost:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    existing_stock = await portfolio_company_collection.find_one({"username": username, "symbol": symbol})

    # if stock already exists for this user, update just some fields
    if existing_stock:
        total_quantity = existing_stock["quantity"] + quantity
        total_spent = (existing_stock["average_buy_price"] * existing_stock["quantity"]) + (current_price * quantity)

        new_avg_price = total_spent / total_quantity
        total_current_value = total_quantity * current_price

        query_filter = {"username": username, "symbol": symbol}
        update_operation = {
            "$set": {
                "quantity": total_quantity,
                "average_buy_price": new_avg_price,
                "total_current_price": total_current_value
            }
        }
        # do the update in the db
        await portfolio_company_collection.update_one(
            query_filter,
            update_operation
        )

    else:
        company_name = await get_company_name_for_symbol(symbol)
        total_current_value = current_price * quantity

        await portfolio_company_collection.insert_one({
            "username": username,
            "symbol": symbol,
            "company_name": company_name,
            "quantity": quantity,
            "average_buy_price": current_price,
            "total_current_price": total_current_value
        })

    #update user amount of virtual money
    query_filter = {"username": username}
    update_operation = {
        "$inc": {
            "virtual_money_balance": -total_cost
        }
    }
    await users_collection.update_one(
        query_filter,
        update_operation
    )

    return {
        "message": "stock purchase recorded succesfully",
        "bought_at_price": current_price,
        "total_cost": total_cost
    }


@portfolio_router.post("/{username}/sell")
async def sell_stock(username: str, symbol: str, quantity: int):
    """Sell stocks and update virtual money balance accordingly."""
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity to be sold must be > 0")

    # Fetch the user's stock
    existing_stock = await portfolio_company_collection.find_one({"username": username, "symbol": symbol})

    if not existing_stock:
        raise HTTPException(status_code=400, detail="Stock not found in the portfolio")

    if quantity > existing_stock["quantity"]:
        raise HTTPException(status_code=400, detail="Cannot sell more than you own")

    # Fetch the current stock price
    current_price = await get_current_stock_price_company(symbol)
    total_sale_value = quantity * current_price
    remaining_quantity = existing_stock["quantity"] - quantity

    # Update virtual money balance **before** modifying the portfolio
    await users_collection.update_one(
        {"username": username},
        {"$inc": {"virtual_money_balance": total_sale_value}}
    )

    if remaining_quantity == 0:
        # Remove stock from portfolio since all shares are sold
        await portfolio_company_collection.delete_one({"username": username, "symbol": symbol})
        return {
            "message": "All shares of this stock removed from the portfolio",
            "sale_value": total_sale_value
        }
    else:
        # Update stock portfolio with new quantity and value
        total_current_value_remained = remaining_quantity * current_price

        await portfolio_company_collection.update_one(
            {"username": username, "symbol": symbol},
            {"$set": {
                "quantity": remaining_quantity,
                "total_current_price": total_current_value_remained
            }}
        )

        return {
            "message": "Stock sale recorded successfully",
            "sale_value": total_sale_value
        }

from unittest.mock import patch, AsyncMock
import pytest
from httpx import AsyncClient

from main import app


@pytest.mark.asyncio
async def test_get_portfolio_for_user_returns_data():
    fake_portfolio = [
        {
            "_id": "mocked_id",
            "username": "testuser",
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "quantity": 10,
            "average_buy_price": 150.0,
            "total_current_price": 1500.0
        }
    ]
    mock_cursor = AsyncMock()
    mock_cursor.to_list = AsyncMock(return_value=fake_portfolio)

    with patch("routers.portfolio_company.portfolio_company_collection.find") as mock_find:
        mock_find.return_value = mock_cursor

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/portfolio/testuser")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert data[0]["symbol"] == "AAPL"



@pytest.mark.asyncio
async def test_buy_stock_successful_purchase():
    mock_user = {"username": "testuser", "virtual_money_balance": 5000.0}
    request_body = {"symbol": "AAPL", "quantity": 2}

    with patch("routers.portfolio_company.get_current_stock_price_company", return_value=100.0), \
            patch("routers.portfolio_company.get_company_name_for_symbol", return_value="Apple Inc."), \
            patch("routers.portfolio_company.users_collection.find_one", new_callable=AsyncMock) as mock_find_user, \
            patch("routers.portfolio_company.portfolio_company_collection.find_one",
                  new_callable=AsyncMock) as mock_find_stock, \
            patch("routers.portfolio_company.portfolio_company_collection.insert_one", new_callable=AsyncMock), \
            patch("routers.portfolio_company.users_collection.update_one", new_callable=AsyncMock):
        mock_find_user.return_value = mock_user
        mock_find_stock.return_value = None

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/portfolio/testuser/buy", json=request_body)
            assert response.status_code == 200
            assert "message" in response.json()
            assert response.json()["total_cost"] == 200.0


@pytest.mark.asyncio
async def test_sell_stock_successfully():
    mock_existing_stock = {
        "_id": "mocked_id",
        "symbol": "AAPL",
        "quantity": 5
    }

    with patch("routers.portfolio_company.get_current_stock_price_company", return_value=100.0), \
            patch("routers.portfolio_company.portfolio_company_collection.find_one",
                  new_callable=AsyncMock) as mock_find_stock, \
            patch("routers.portfolio_company.users_collection.update_one", new_callable=AsyncMock), \
            patch("routers.portfolio_company.portfolio_company_collection.update_one", new_callable=AsyncMock):
        mock_find_stock.return_value = mock_existing_stock

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/portfolio/testuser/sell", json={"symbol": "AAPL", "quantity": 2})
            assert response.status_code == 200
            assert "sale_value" in response.json()
            assert response.json()["sale_value"] == 200.0

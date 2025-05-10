import pytest
from httpx import AsyncClient
from main import app
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_get_stock_data_valid():
    mock_provider = MagicMock()
    mock_provider.get_stock_data.return_value = {
        "provider": "yahoo", "symbol": "AAPL", "latest_price": 150.0
    }

    with patch("routers.stocks.get_provider", return_value=mock_provider):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/stocks/stock/yahoo/AAPL")
            assert response.status_code == 200
            assert response.json()["symbol"] == "AAPL"

@pytest.mark.asyncio
async def test_get_stocks_data_for_symbol_substring_live():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/stocks/yahoo", params={"symbol_substr": "AMZN"})
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Check the structure of one item
        first = data[0]
        assert "provider" in first
        assert "company_name" in first
        assert "symbol" in first
        assert "latest_price" in first

        # Optional: Check if "AMZN" is among the symbols
        symbols = [item["symbol"] for item in data]
        assert any("AMZN" in s for s in symbols)

@pytest.mark.asyncio
async def test_get_monthly_stock_data():
    mock_provider = MagicMock()
    mock_provider.get_monthly_close_prices.return_value = {
        "symbol": "AAPL",
        "monthly_prices": {"2023-01": {"Close": 150.0}}
    }

    with patch("routers.stocks.get_provider", return_value=mock_provider):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/stocks/monthly/yahoo/AAPL")
            assert response.status_code == 200
            assert response.json()["symbol"] == "AAPL"

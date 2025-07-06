# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    virtual_money_balance = Column(Float, default=15000.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PortfolioCompany(Base):
    __tablename__ = "portfolio_company"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username"))
    symbol = Column(String)
    company_name = Column(String)
    quantity = Column(Integer)
    average_buy_price = Column(Float)
    total_current_value = Column(Float)

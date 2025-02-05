from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data.db.database import init_db, close_mongo

from routers.auth import router as auth_router
from routers.stocks import stocks_router
from routers.portfolio_company import portfolio_router

app = FastAPI(title="User Portfolio API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    await init_db()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo()

app.include_router(auth_router)
app.include_router(stocks_router)
app.include_router(portfolio_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Finance App"}
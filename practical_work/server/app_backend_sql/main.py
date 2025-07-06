from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from data.database.database import mongodb
from routers.stock_data_router import stock_data_router
from routers.authentication_router import authentication_router
from routers.portfolio_router import portfolio_router
from routers.quiz_router import quiz_router


from data.database.db import engine
from data.models.models_sql_alchemy import Base

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("DATABASE INITED")
# async def startup_db():
#     await mongodb.connect()

@app.on_event("shutdown")
def shutdown_db():
    print("App shutdown")
    # mongodb.close()

app.include_router(authentication_router)
app.include_router(stock_data_router)
app.include_router(portfolio_router)
app.include_router(quiz_router)

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}

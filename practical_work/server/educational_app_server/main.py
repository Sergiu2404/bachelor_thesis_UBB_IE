from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data.database.database import mongodb
from routers.stock_data_router import stock_data_router
from routers.authentication_router import authentication_router
from routers.portfolio_router import portfolio_router
from routers.quiz_router import quiz_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db():
    await mongodb.connect()

@app.on_event("shutdown")
def shutdown_db():
    mongodb.close()

# @app.on_event("startup")
# async def startup_db_client():
#     await init_db()
#
# @app.on_event("shutdown")
# async def shutdown_db_client():
#     await close_mongo()

app.include_router(authentication_router)
app.include_router(stock_data_router)
app.include_router(portfolio_router)
app.include_router(quiz_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

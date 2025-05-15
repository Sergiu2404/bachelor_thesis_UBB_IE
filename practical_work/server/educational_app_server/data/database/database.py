from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config.config import MONGO_URI, DB_NAME

class MongoDB:
    def __init__(self, uri: str, db_name: str):
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[db_name]
        self.users = self._db.users_collection
        self.portfolio = self._db.portfolio_company_collection

    async def connect(self):
        try:
            await self._client.admin.command('ping')
            await self.users.create_index("username", unique=True)
            await self.users.create_index("email", unique=True)
            print("MongoDB connected and indexes ensured.")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print("MongoDB connection failed:", e)
            raise

    def close(self):
        self._client.close()

mongodb = MongoDB(MONGO_URI, DB_NAME)

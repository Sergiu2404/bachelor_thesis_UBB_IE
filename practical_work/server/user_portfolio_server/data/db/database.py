from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

MONGO_URI = "mongodb+srv://sergiu_goian:SergiuMONGODB2404!@cluster0.vktc6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize at module level
client = AsyncIOMotorClient(MONGO_URI)
db = client.user_portfolio_db
users_collection = db.users_collection


async def init_db():
    try:
        # Test the connection
        await client.admin.command('ping')
        print('Successfully connected to MongoDB')

        # Create indexes
        await users_collection.create_index("username", unique=True)
        await users_collection.create_index("email", unique=True)

        print("Database initialized successfully")
        return True

    except ServerSelectionTimeoutError:
        print("Could not connect to MongoDB. Please check if the URI is correct and the server is running.")
        raise
    except ConnectionFailure:
        print("Failed to connect to MongoDB")
        raise
    except Exception as e:
        print(f"An error occurred while initializing database: {str(e)}")
        raise


async def close_mongo():
    if client:
        client.close()
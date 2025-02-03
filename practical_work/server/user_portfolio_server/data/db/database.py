# from motor.motor_asyncio import AsyncIOMotorClient
#
# MONGO_URI = "mongodb+srv://sergiu_goian:SergiuMONGODB2404!@cluster0.vktc6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
#
# client = AsyncIOMotorClient(MONGO_URI)
# db = client["user-portfolio-thesis"]
# users_collection = db["users"]

from pymongo.mongo_client import MongoClient

MONGO_URI = "mongodb+srv://sergiu_goian:SergiuMONGODB2404!@cluster0.vktc6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(MONGO_URI)

db = client.user_portfolio_db

users_collection = db["users_collection"]

try:
    client.admin.command('ping')
    print('Pinged mongo env')
except Exception as ex:
    print(ex)



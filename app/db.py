from pymongo import MongoClient
import os

def get_db():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
    client = MongoClient(mongo_uri)
    return client["Network"]  # Nombre exacto de la base de datos

def save_results(db, collection_name, data):
    if data:
        db[collection_name].insert_many(data)
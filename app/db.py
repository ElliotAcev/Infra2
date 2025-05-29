from pymongo import MongoClient

def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    return client["Network"]  # Asegúrate que sea exactamente "Network" como ya está creada

def save_results(db, collection_name, data):
    if data:
        db[collection_name].insert_many(data)
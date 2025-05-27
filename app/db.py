from pymongo import MongoClient

def get_db():
    return MongoClient("mongodb://localhost:27017")["Network"]

def get_collecttion(nombre):
    return get_db()[nombre]
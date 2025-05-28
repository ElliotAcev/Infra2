import pymongo 

def get_db():
    cliente = pymongo.MongoClient("mongodb://localhost:27017/")
    db = cliente["NetWork"]
    return db



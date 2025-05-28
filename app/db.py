import pymongo 

def get_db():
    cliente = pymongo.MongoClient("mongodb://localhost:27017/")
    db = cliente["NetWork"]
    return db

def save_results(db, coleccion, datos):
    try:
        resultado = db[coleccion].insert_many(datos)
        print("Guardados {len(resultado.inserted_ids)} documentos en la colección '{Anomalos}'.")
    except Exception as e:
        print(f"error al guardar: ", e)


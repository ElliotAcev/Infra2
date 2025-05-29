import torch
import torch.nn as nn
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim): # inicializa el autoencoder con una capa de entrada y una capa de salida
        super().__init__() # llama al constructor de la clase padre nn.Module
        self.encoder = nn.Sequential( # define la arquitectura del autoencoder
            nn.Linear(input_dim, 16), # la primera capa lineal que reduce la dimensionalidad de la entrada
            nn.ReLU(), # función de activación ReLU
            nn.Linear(16, 8), # la segunda capa lineal que reduce aún más la dimensionalidad
            nn.ReLU()# función de activación ReLU
        )   #sequential encadena varias capas
        self.decoder = nn.Sequential( # define la parte de decodificación del autoencoder
            nn.Linear(8, 16), # la primera capa lineal que aumenta la dimensionalidad de la representación codificada
            nn.ReLU(), # función de activación ReLU
            nn.Linear(16, input_dim), # la segunda capa lineal que reconstruye la entrada original
        )
    def forward(self, x): #funcion de paso hacia adelante que recibe un tensor x
        return self.decoder(self.encoder(x))  #paso hacia adelante, primero codifica y luego decodifica

def save_model(model, path = 'autoencoder.pth'): # funcion para guardar el modelo
    torch.save(model.state_dict(), path)  # guarda el estado del modelo en un archivo especificado por path

def load_model(input_dim, path = 'autoencoder.pth'): # funcion para cargar el modelo
    model = Autoencoder(input_dim)  # crea una instancia del modelo Autoencoder con la dimensión de entrada especificada
    model.load_state_dict(torch.load(path))  # carga el estado del modelo desde el archivo especificado por path
    return model  # devuelve el modelo cargado

def train_autoencoder(model, data, epochs = 100, lr = 1e-3): # funcion para entrenar el autoencoder
    opti = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for epoch in range(epochs): # bucle de entrenamiento

        model.train() #  pone el modelo en modo entrenamiento
        output = model(data) # pasa los datos de entrada por el modelo para obtener la salida
        loss = crit(output, data) # calcula la pérdida entre la salida del modelo y los datos de entrada

        opti.zero_grad() # pone a cero los gradientes del optimizador
        loss.backward() # calcula los gradientes de la pérdida respecto a los pesos del modelo
        opti.step() # actualiza los pesos del modelo

        if epoch % 10 == 0: # imprime la pérdida cada 10 épocas
            print(f'Epoch {epoch}, Loss: {loss.item():.5f}') # imprime la pérdida cada 10 épocas
    return model # devuelve el modelo entrenado

def anomaly_detection(model, data, umbral = None): #funcion para detectar anomalías
    model.eval()

    with torch.no_grad(): # desactiva el cálculo de gradientes para ahorrar memoria y tiempo
        rec = model(data) # reconstruye los datos de entrada
        error = torch.mean((data - rec)**2, dim = 1).cpu().numpy() # calcula el error cuadrático medio 

    if umbral is None: # si no se ha especificado un umbral, se calcula el percentil 95 del error
        umbral = np.percentile(error, 95) # umbral por defecto es el percentil 95 del error

    return error, error > umbral, umbral # devuelve el error, un booleano indicando si es una anomalía y el umbral utilizado

ModelP = 'autoencoder.pth' # ruta del modelo guardado 
scalerP = 'scaler.pkl' # ruta del scaler guardado

def save_scaler(scaler, path = scalerP): # funcion para guardar el scaler
    joblib.dump(scaler, path)  # guarda el scaler en un archivo especificado por path
def load_scaler(path = scalerP): # funcion para cargar el scaler
    return joblib.load(path)  # carga el scaler desde el archivo especificado por path

def proc_and_train(df): # Procesa los datos y entrena el modelo de autoencoder
    if df.empty: # verifica si el dataframe está vacío  
        raise ValueError("El dataframe está vacío. Por favor, proporciona datos válidos.") # verifica si el dataframe está vacío
    
    scaler = MinMaxScaler() # crea una instancia del escalador MinMaxScaler
    X = scaler.fit_transform(df) # ajusta el escalador a los datos y transforma los datos a un rango entre 0 y 1
    X_tensor = torch.tensor(X, dtype=torch.float32) # convierte los datos a un tensor de PyTorch de tipo float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # verifica si hay una GPU disponible y usa CUDA si es así, de lo contrario usa CPU
    X_tensor = X_tensor.to(device) # mueve el tensor a la GPU si está disponible

    # Entrena desde cero para esta sesión de análisis
    model = Autoencoder(X.shape[1]) #  crea una instancia del modelo Autoencoder con la dimensión de entrada igual al número de columnas del dataframe
    model.to(device) # mueve el modelo a la GPU si está disponible
    model = train_autoencoder(model, X_tensor) # entrena el modelo con los datos de entrada

    save_model(model, ModelP)
    save_scaler(scaler, scalerP)

    error, anomaly, umbral = anomaly_detection(model, X_tensor) # detecta anomalías en los datos de entrada utilizando el modelo entrenado

    df_out = df.copy()
    df_out['error'] = error
    df_out['anomaly'] = anomaly

    return df_out, model, scaler, error

def retrain_model(df):
    if df.empty:
        raise ValueError("El dataframe está vacío. Por favor, proporciona datos válidos.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(scalerP):
        scaler = load_scaler(scalerP)

        # Detectar cambio en cantidad de columnas
        if scaler.n_features_in_ != df.shape[1]:
            print("⚠️ Número de columnas diferente. Reentrenando desde cero.")
            scaler = MinMaxScaler()
            X = scaler.fit_transform(df)
            save_scaler(scaler, scalerP)

            model = Autoencoder(X.shape[1])
            model.to(device)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            model = train_autoencoder(model, X_tensor)
            save_model(model, ModelP)

            return model, scaler
        else:
            X = scaler.transform(df)
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df)
        save_scaler(scaler, scalerP)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    if os.path.exists(ModelP):
        model = load_model(X.shape[1], ModelP)
        model.to(device)
    else:
        model = Autoencoder(X.shape[1])
        model.to(device)

    model = train_autoencoder(model, X_tensor)
    save_model(model, ModelP)

    return model, scaler

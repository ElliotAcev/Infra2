import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim): # inicializa el autoencoder con una capa de entrada y una capa de salida
        super().__init__() # llama al constructor de la clase padre nn.Module
        self.encoder = nn.Sequential( # define la arquitectura del autoencoder
            nn.linear(input_dim, 16), # la primera capa lineal que reduce la dimensionalidad de la entrada
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

def train_autoencoder(model, data, epochs = 50, lr = 1e-3): # funcion para entrenar el autoencoder
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

def anomaly_detection(model, data, umbral = None): #funcion para detectar anomalías
    model.eval()

    with torch.no_grad(): # desactiva el cálculo de gradientes para ahorrar memoria y tiempo
        rec = model(data) # reconstruye los datos de entrada
        error = torch.mean((data - rec)**2, dim = 1).numpy() # calcula el error cuadrático medio 

    if umbral is None: # si no se ha especificado un umbral, se calcula el percentil 95 del error
        umbral = np.percentile(error, 95) # umbral por defecto es el percentil 95 del error

    return error, error > umbral, umbral # devuelve el error, un booleano indicando si es una anomalía y el umbral utilizado
    
def proc_and_train(df):
    scaler = MinMaxScaler() #para normalizar los datos entre 0 y 1
    X = scaler.fit_transform(df) # normaliza los datos entre 0 y 1
    X_tensor = torch.tensor(X, dtype=torch.float32) # convierte el dataframe a un tensor de PyTorch

    model = Autoencoder(X.shape[1]) # crea el modelo
    model = train_autoencoder(model, X_tensor) # entrena el modelo

    error, anomaly, umbral = anomaly_detection(model, X_tensor) # detecta anomalías

    df_out = df.copy() # crea una copia del dataframe original
    df_out['error']= error  # filtra el dataframe original para obtener las anomalías
    df_out['anomaly'] = anomaly  # añade una columna con las anomalías

    return df_out, model, scaler

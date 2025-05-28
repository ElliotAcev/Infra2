import streamlit as st
import pandas as pd
from ml import proc_and_train
from db import get_db, save_results

st.set_page_config(page_title = "Anomaly Network Detection", page_icon = "🔍", layout = "wide")
st.title("Anomaly Network Detection")

st.markdown("Sube un archivo csv para analizarlo")
archivo = archivos = st.file_uploader("Sube archivos CSV", type=["csv"], accept_multiple_files=True)

if archivo is not None:
    try:
        dfs = [pd.read_csv(f, header=None) for f in archivos]
        df = pd.concat(dfs, ignore_index=True)
        st.subheader("Vista Previa")
        st.dataframe(df.head())

        

        columnas_info = {
        0: "ID de trayectoria",
        1: "Potencia de la trayectoria (Path Gain)",
        2: "Retardo (Delay) en ns",
        3: "Ángulo de llegada (AoA) en grados",
        4: "Ángulo de salida (AoD) en grados",
        5: "Frecuencia o canal",
        6: "Distancia al receptor (m)",
        7: "Distancia al transmisor (m)",
        8: "Identificador de entorno",
        9: "Parámetro técnico adicional",
        'error': "Error de reconstrucción (usado para detectar anomalías)",
        'anomaly': "Etiqueta de anomalía (True si es anómalo)"
        } 
        with st.spinner("Procesando..."):
            df_result, modelo, umbral = proc_and_train(df)
            df_anom = df_result[df_result["anomaly"] == True]
            df_norm = df_result[df_result["anomaly"] == False]
            
            st.write("🧾 Columnas del dataset procesado:")
            st.write(df_result.columns.tolist())

        with st.expander("ℹ️ ¿Qué significa cada columna?"):
            for col in df_result.columns:
                descripcion = columnas_info.get(col, "Descripción no disponible.")
                st.markdown(f"**{col}:** {descripcion}")

       
        opcion = st.radio(
        "Selecciona qué registro desea ver",
        ("Todos", "Solo anómalos", "Solo normales")
        )

# Mostrar según opción
        if opcion == "Todos":
            st.subheader(" Todos los registros")
            st.dataframe(df_result)
        elif opcion == "Solo anómalos":
            st.subheader("⚠️ Anomalías detectadas")
            st.dataframe(df_anom)
        elif opcion == "Solo normales":
            st.subheader("✅ Registros normales")
            st.dataframe(df_norm)

        st.metric(label="Anomalías detectadas", value=len(df_anom))

        if st.button("Cargar en Base de Datos"):
            db = get_db()
            save_results(db, "Anomalos", df_result.to_dict(orient = 'records'))
            st.success("Resultados guardados en la base de datos.")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
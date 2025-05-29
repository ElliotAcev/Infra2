import streamlit as st
import pandas as pd
import numpy as np
from ml import proc_and_train, load_model, retrain_model, load_scaler, anomaly_detection,torch
from db import get_db, save_results
import plotly.express as px


st.set_page_config(page_title="Anomaly Network Detection", page_icon="🔍", layout="wide")
st.title("Anomaly Network Detection")

st.markdown("Sube un archivo csv para analizarlo")
archivos = st.file_uploader("Sube archivos CSV", type=["csv"], accept_multiple_files=True)

if archivos:
    try:
        dfs = [pd.read_csv(f, header=None) for f in archivos]
        df = pd.concat(dfs, ignore_index=True)
        st.subheader("Vista Previa")
        st.dataframe(df.head())

        # Procesamiento inicial
        with st.spinner("Procesando datos..."):
            df_result, modelo, umbral_def, errores = proc_and_train(df)
            df_result.columns = df_result.columns.map(str)
            df_result["error"] = errores

        st.write("Máximo error:", np.max(errores))
        st.write("Mínimo error:", np.min(errores))
        st.write("Promedio error:", np.mean(errores))

        # Slider para cambiar el umbral
        Slider_umbral = st.slider("Ajusta el umbral de detección de anomalías",
                                  float(np.min(errores)),
                                  float(np.max(errores)),
                                  float(np.percentile(errores, 95)))

        df_result["anomaly"] = errores > Slider_umbral
        df_anom = df_result[df_result["anomaly"]]
        df_norm = df_result[~df_result["anomaly"]]

        df_result.columns = df_result.columns.map(str)  # Asegura que todas las columnas sean string
        # Descripción de columnas
        columnas_info = {
            '0': "ID de trayectoria",
            '1': "Potencia de la trayectoria (Path Gain)",
            '2': "Retardo (Delay) en ns",
            '3': "Ángulo de llegada (AoA) en grados",
            '4': "Ángulo de salida (AoD) en grados",
            '5': "Frecuencia o canal",
            '6': "Distancia al receptor (m)",
            '7': "Distancia al transmisor (m)",
            '8': "Identificador de entorno",
            '9': "Parámetro técnico adicional",
            'error': "Error de reconstrucción",
            'anomaly': "Etiqueta de anomalía"
        }

        with st.expander("¿Qué significa cada columna?"):
            for col in df_result.columns:
                desc = columnas_info.get(col, 'Descripción no disponible')
                st.markdown(f"**{col}:** {desc}")
        
        st.metric(label="Anomalías detectadas", value=len(df_anom))
        # Mostrar datos según elección
        opcion = st.radio("Selecciona qué registros deseas ver:",
                          ("Todos", "Solo anómalos", "Solo normales"))
        if opcion == "Todos":
            st.subheader("Todos los registros")
            st.dataframe(df_result)
        elif opcion == "Solo anómalos":
            st.subheader("⚠️ Anomalías detectadas")
            st.dataframe(df_anom)
        elif opcion == "Solo normales":
            st.subheader("✅ Registros normales")
            st.dataframe(df_norm)

        

        # Guardar resultados en MongoDB
        opcion_guardado = st.radio("¿Qué tipo de datos deseas guardar en MongoDB?",
                                   ["Anómalos", "Normales"])
        if opcion_guardado == "Anómalos":
            if st.button("Guardar anómalos en MongoDB"):
                db = get_db()
                if not df_anom.empty:
                    db.Anomalos.insert_many(df_anom.to_dict(orient="records"))
                    st.success("✅ Datos anómalos guardados.")
                else:
                    st.warning("⚠️ No hay registros anómalos para guardar.")
        elif opcion_guardado == "Normales":
            if st.button("Guardar normales en MongoDB"):
                db = get_db()
                if not df_norm.empty:
                    db.Normales.insert_many(df_norm.to_dict(orient="records"))
                    st.success("✅ Datos normales guardados.")
                else:
                    st.warning("⚠️ No hay registros normales para guardar.")

        # Reentrenar modelo con datos guardados
        st.markdown("---")
        st.subheader("🔁 Reentrenar modelo con datos guardados")
        reentrenar_opcion = st.radio("Selecciona los datos para reentrenar:",["Normales", "Anómalos", "Ambos"])

        if st.button("Reentrenar Autoencoder con datos guardados"):
            try:
                db = get_db()
                df_normales = pd.DataFrame(list(db.Normales.find()))
                df_anomalos = pd.DataFrame(list(db.Anomalos.find()))

                if reentrenar_opcion == "Normales":
                    df_entrenar = df_normales
                elif reentrenar_opcion == "Anómalos":
                    df_entrenar = df_anomalos
                else:
                    df_entrenar = pd.concat([df_normales, df_anomalos], ignore_index=True)

                df_entrenar = df_entrenar.drop(columns=["_id", "error", "anomaly"], errors="ignore")

                if not df_entrenar.empty:
                    try:
                        # Antes del reentrenamiento
                        errores_antes = errores.copy()
                        df_result_antes = df_result.copy()
                        model, scaler = retrain_model(df_entrenar)

                        # Aplicar el modelo reentrenado a los mismos datos
                        X_scaled = scaler.transform(df_entrenar)
                        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(model.encoder[0].weight.device)

                        error, anomaly, umbral = anomaly_detection(model, X_tensor)

                        # Actualiza el dataframe con los nuevos resultados
                        df_result = df_entrenar.copy()
                        df_result["error"] = error
                        df_result["anomaly"] = anomaly

                        st.success("Modelo reentrenado exitosamente")
                        st.dataframe(df_result)

                    except Exception as e:
                        st.error(f"❌ Error al reentrenar : {e}.")

                else:
                    st.warning("⚠️ No hay datos para reentrenar.")
            except Exception as e:
                st.error(f"❌ Error al reentrenar: {e}")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

with st.sidebar:
    st.header("📊 Comparación del Modelo")

    if 'errores' in locals() and 'df_result' in locals():
        fig_error = px.histogram(df_result, x="error", nbins=50, title="Distribución de Errores")
        st.plotly_chart(fig_error, use_container_width=True)

        fig_anom = px.bar(x=["Anómalos", "Normales"],
                          y=[df_result["anomaly"].sum(), (~df_result["anomaly"]).sum()],
                          title="Conteo de Registros")
        st.plotly_chart(fig_anom, use_container_width=True)

if 'errores_antes' in locals() and 'df_result_antes' in locals():
    st.subheader("📉 Comparación Antes vs Después")

    st.markdown("**Antes del reentrenamiento**")
    fig1 = px.histogram(df_result_antes, x="error", nbins=50)
    st.plotly_chart(fig1)

    st.markdown("**Después del reentrenamiento**")
    fig2 = px.histogram(df_result, x="error", nbins=50)
    st.plotly_chart(fig2)

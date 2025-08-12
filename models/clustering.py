import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def clustering(df: pd.DataFrame):
    st.header("Clustering (Agrupamiento)")

    # Variables numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) < 2:
        st.warning("Se necesitan al menos dos columnas numéricas para hacer clustering.")
        return

    # Selección de variables
    variables = st.multiselect("Selecciona variables para el clustering", columnas_numericas, default=list(columnas_numericas[:2]))

    if len(variables) < 2:
        st.info("Selecciona al menos dos variables.")
        return

    # Escalar datos
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df[variables])

    # Selección de número de clusters
    k = st.slider("Número de clusters (K)", 2, 10, 3)

    # Modelo K-Means
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = modelo.fit_predict(datos_escalados)

    # Mostrar centros
    st.subheader("Centroides de los clusters")
    centroides = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=variables)
    st.dataframe(centroides)

    # Visualización
    st.subheader("Visualización de Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(x=variables[0], y=variables[1], hue='cluster', data=df, palette='tab10', ax=ax)
    ax.set_title("Clusters detectados")
    st.pyplot(fig)

    # Guardar resultado en session_state
    st.session_state.df_actual = df

    # Descargar datos con clusters
    st.download_button(
        label="📥 Descargar datos con clusters",
        data=df.to_csv(index=False),
        file_name="datos_clusters.csv",
        mime="text/csv"
    )

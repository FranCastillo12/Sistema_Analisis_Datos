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
from sklearn.decomposition import PCA

def clustering(df: pd.DataFrame):
    
    with st.expander("Clustering"): 
        
        st.header("Clustering (Agrupamiento)")

                            
        # Variables numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        if len(columnas_numericas) < 2:
            st.warning("Se necesitan al menos dos columnas numéricas para hacer clustering.")
            return
        col1, col2 = st.columns(2)
        with col1:
            variables = st.multiselect("Selecciona variables para el clustering", columnas_numericas, default=list(columnas_numericas[:2]))
        with col2:
            # Selección de número de clusters
            #permite experimentar y visualizar cómo se agrupan tus datos según la cantidad de clusters elegida.
            k = st.slider("Número de clusters (K)", 2, 10, 3)

        if len(variables) < 2:
            st.info("Selecciona al menos dos variables.")
            return

        # Escalar datos
        #Se normalizan las variables donde media = 0 y desviación estándar = 1.
        scaler = StandardScaler()
        datos_escalados = scaler.fit_transform(df[variables])

        # Se entrena el Modelo K-Means
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = modelo.fit_predict(datos_escalados)
        
        
        # Crear copia del DataFrame para no modificar el original
        df_resultado = df.copy()
        df_resultado['cluster'] = clusters.astype(str)  # convierto a str para mejor manejo en gráficos


        # Mostrar centroides en escala original
        st.subheader("Centroides de los clusters")
        centroides_original = pd.DataFrame(
            scaler.inverse_transform(modelo.cluster_centers_),
            columns=variables
        )
        centroides_original['cluster'] = centroides_original.index.astype(str)
        st.dataframe(centroides_original)


        # Insight automático
        st.markdown("**Insight automático:**")

        # Tamaño de cada cluster
        tamaños = df_resultado['cluster'].value_counts().sort_index()
        for cluster_id, count in tamaños.items():
            porcentaje = count / len(df_resultado) * 100
            st.markdown(f"- El **Cluster {cluster_id}** contiene **{count}** observaciones ({porcentaje:.1f}%).")

        # Variables más diferenciadoras (basado en varianza entre centroides)
        centroides_std = centroides_original.drop(columns='cluster').std()
        variable_top = centroides_std.idxmax()
        st.markdown(f"- La variable que más diferencia a los clusters es **{variable_top}**, según la dispersión entre centroides.")

        # Sugerencia de uso 
        st.markdown("- Estos grupos pueden representar distintos perfiles o segmentos. Considera analizar cada cluster por separado para descubrir patrones únicos.")
        
        
        

        # Visualización de clusters
        st.subheader("Visualización de Clusters")

        fig, ax = plt.subplots(figsize=(8,6))

        if len(variables) == 2:
            # Si solo hay dos variables, graficar directamente
            sns.scatterplot(
                x=variables[0], y=variables[1],
                hue='cluster',
                palette='tab10',
                data=df_resultado,
                s=80,
                alpha=0.7,
                ax=ax
            )   
            ax.set_title("Clusters detectados")
            ax.legend(title="Cluster")
            ax.grid(True)

        else:
            # usar PCA para reducción a 2D
            pca = PCA(n_components=2, random_state=42)
            componentes = pca.fit_transform(datos_escalados)
            df_pca = pd.DataFrame(componentes, columns=['PC1', 'PC2'])
            df_pca['cluster'] = clusters.astype(str)

            sns.scatterplot(
                x='PC1', y='PC2',
                hue='cluster',
                palette='tab10',
                data=df_pca,
                s=80,
                alpha=0.7,
                ax=ax
            )
            ax.set_title("Clusters detectados (PCA 2D)")
            ax.legend(title="Cluster")
            ax.grid(True)

        st.pyplot(fig)

        # Guardar resultado en session_state para uso posterior
        st.session_state.df_actual = df_resultado
        
        st.subheader("Estadísticas descriptivas por cluster")
        

        # Descargar datos con clusters
        st.download_button(
            label="📥 Descargar datos con clusters",
            data=df.to_csv(index=False),
            file_name="datos_clusters.csv",
            mime="text/csv"
        )   


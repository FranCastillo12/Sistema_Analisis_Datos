import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


#Funcion para realizar el analisis del dataframe ya limpio
def analisis_exploratorio(df: pd.DataFrame):
    
    st.header("Exploración y Visualización")

    # Separar numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns


    st.subheader("Estadísticas generales")
    st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    st.dataframe(df.describe())

    st.subheader("Vista general")
    #Se muestra una vista general del dataframe
    st.write(df.head(10).iloc[:, :20])

    #Mostrar los tipos de datos
    st.subheader("Tipos de datos")
    tipos_exactos = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de dato': df.dtypes.astype(str)
        })
    st.dataframe(tipos_exactos)
   

    #Si las columbas numericas son mayores a 0
    if len(columnas_numericas) > 0:

        #Se creab histogramas
        st.subheader("Histogramas")
        col1, col2 = st.columns(2)
        with col1:
            variable_num = st.selectbox("Selecciona variable numérica", columnas_numericas)
        with col2:
            bins = st.slider("Cantidad de bins", max_value=100, value=10)

        fig, ax = plt.subplots()
        sns.histplot(df[variable_num], bins=bins, kde=True, ax=ax)
        ax.set_title(f"Distribución de {variable_num}")
        st.pyplot(fig)

        st.subheader("Boxplot por variable")
        var_box = st.selectbox("Selecciona variable numérica", columnas_numericas, key="box")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[var_box], ax=ax)
        ax.set_title(f"Boxplot de {var_box}")
        st.pyplot(fig)  










     #Si las columbas numericas son mayores o iguales a 2
    if len(columnas_numericas) >= 2:

        #Se muestran matriz de correlacion
        st.subheader("Matriz de correlación")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap="YlGnBu", vmax=1, vmin=-1)
        st.pyplot(fig)

        st.subheader("Correlaciones destacadas")
        cor_matrix = df[columnas_numericas].corr().abs()

        # Eliminar duplicados simétricos
        cor_matrix_values = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
        cor_matrix_values = cor_matrix_values.stack().reset_index()
        cor_matrix_values.columns = ['Variable 1', 'Variable 2', 'Correlación']

        # Filtrar correlaciones fuertes
        umbral = 0.7
        correlaciones_fuertes = cor_matrix_values[cor_matrix_values['Correlación'] >= umbral]

        # Mostrar interpretaciones
        if not correlaciones_fuertes.empty:
            st.write("🔍 Variables fuertemente relacionadas:")
            for _, row in correlaciones_fuertes.iterrows():
                v1 = row['Variable 1']
                v2 = row['Variable 2']
                r = row['Correlación']
                if r > 0.85:
                    interpretacion = "prácticamente lineales"
                elif r > 0.7:
                    interpretacion = "fuertemente relacionadas"
                else:
                    interpretacion = "moderadamente relacionadas"
                st.markdown(f"- **{v1}** y **{v2}** están {interpretacion} (r = {r:.2f})")
        else:
            st.info("No se encontraron correlaciones fuertes entre las variables numéricas.")


    


        


    #Columnas categoricas
    if len(columnas_categoricas) > 0:
        #Se muestra grafico de distribucion de variavles
        st.subheader("Distribución de variables categóricas")
        variable_cat = st.selectbox("Selecciona una variable categórica", columnas_categoricas)

        fig, ax = plt.subplots()
        df[variable_cat].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Distribución de {variable_cat}")
        st.pyplot(fig)




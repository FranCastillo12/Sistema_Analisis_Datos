import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats



# Funcion para subir los archivos
def cargar_archivos(archivos):
    for archivo in archivos:
        if archivo.name not in st.session_state.data_diccio:
            try:
                if archivo.name.endswith('.csv'):
                    #leer archivo cvs
                    df = pd.read_csv(archivo)
                else:
                    #leer archivo exel
                    df = pd.read_excel(archivo)

                #Se guardan los dataframe en el diccionarios de los archivos    
                st.session_state.data_diccio[archivo.name] = df
                st.success(f"Archivo {archivo.name} cargado con éxito.")
            except Exception as e:
                st.error(f"No se pudo leer el archivo {archivo.name}: {e}")



#Funcion para mostrar los archivos subidos
def mostrar_archivos():
    #valida si el diccionario tiene archivos guardados
    if st.session_state.data_diccio:
        st.subheader("Archivos cargados")
        #se recorre el diccionario
        for archivo_nombre in st.session_state.data_diccio:
            #se imprime el nombre del archivo
            st.write(f"- {archivo_nombre}")


        #select para que el usuario seleccione los archivos de los subidos 
        archivos_seleccionados = st.multiselect(
            "Selecciona uno o más archivos para analizar / combinar",
            options=list(st.session_state.data_diccio.keys())
        )

        #Si se seleccionan archivos entra en el if
        if archivos_seleccionados:

            #Solo se selecciona un archivo
            if len(archivos_seleccionados) == 1:
                #En df se guarda los datos que st.session_state.data_diccio[0] 
                df = st.session_state.data_diccio[archivos_seleccionados[0]]

                st.subheader(f"Vista previa de {archivos_seleccionados[0]}")
                #Se llama para la limpieza de datos
                df = limpieza_basica(df)
                #Se muestran los datos en un dataframe
                st.dataframe(df.head())
                               
                #Se guardan los datos limptos para el uso en los otras opciones
                st.session_state["df_actual"] = df


            #Se selecciono mas de un archivo
            else:

                st.subheader(f"Archivos combinados ({len(archivos_seleccionados)})")
                #Se crea una lista de dataframes
                dfs = [st.session_state.data_diccio[f] for f in archivos_seleccionados]
                #Se verifica si los dataframes tienen las mismas columnas
                columnas_primero = set(dfs[0].columns)
                mismo_esquema = all(set(df.columns) == columnas_primero for df in dfs)
                #Si tienen el mismo esquema
                if mismo_esquema:
                    df_comb = pd.concat(dfs, ignore_index=True)
                    st.dataframe(df_comb.head())
                    st.write("### Tipos exactos de datos por columna:")
                    tipos_exactos = pd.DataFrame({
                    'Columna': df.columns,
                    'Tipo de dato': df.dtypes.astype(str)
                    })

                    st.dataframe(tipos_exactos)
                    st.session_state["df_actual"] = df_comb

                else:
                    st.warning("Los archivos seleccionados no tienen la misma estructura de columnas y no se pueden combinar.")


#Funcion para limpiar los dataframe escogidos
def limpieza_basica(df: pd.DataFrame) -> pd.DataFrame:
    st.write("### Limpieza")

    # Mostrar shape original
    st.write(f"Dimensiones originales: {df.shape}")

    #Eliminar filas con datos nulos
    df = df.dropna()

    # Eliminar duplicados
    df.duplicated()


    #Recorre las columnas que tienen datos object o string
    for col in df.select_dtypes(include=['object', 'string']).columns:
        #Pasa los daatos de cada columna a minuscula
        df.loc[:, col] = df[col].str.lower().str.strip()
   
    #se imprimen los dataframen ya limpios
    st.write(f"Dimensiones después de limpieza: {df.shape}")
    return df

#Funcion para realizar el analisis del dataframe ya limpio
def analisis_exploratorio(df: pd.DataFrame):

    st.subheader("Dimensiones")
    #Se muestra cuales son las dimensiones del dataframe
    st.write(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

    st.subheader("Vista general del dataset")
    #Se muestra una vista general del dataframe
    st.write(df.head(10).iloc[:, :20])

    #Mostrar los tipos de datos
    st.subheader("Tipos de datos")
    tipos_exactos = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de dato': df.dtypes.astype(str)
        })
    st.dataframe(tipos_exactos)
   


    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())

    # Separar numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns

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

     #Si las columbas numericas son mayores o iguales a 2
    if len(columnas_numericas) >= 2:

        #Se muestran matriz de correlacion
        st.subheader("Matriz de correlación")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap="YlGnBu", vmax=1, vmin=-1)
        st.pyplot(fig)


    #Columnas categoricas
    if len(columnas_categoricas) > 0:
        #Se muestra grafico de distribucion de variavles
        st.subheader("Distribución de variables categóricas")
        variable_cat = st.selectbox("Selecciona una variable categórica", columnas_categoricas)

        fig, ax = plt.subplots()
        df[variable_cat].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Distribución de {variable_cat}")
        st.pyplot(fig)

#Funcion para realizar outliers
def outliers(df: pd.DataFrame):

    #Se crea una copia del dataframe que se esta analizando
    df = df.copy()

    #Se guardan las columnas que tienen datos numericos
    numericos = df.select_dtypes(include=[np.number])

    if numericos.empty:
        st.warning("No hay columnas numéricas en el archivo.")
    else:

        for columnas in numericos.columns:

            #uso de Z-Score
            z_cores = np.abs(stats.zscore(numericos[columnas]))
            df[f'{columnas}_z_outlier'] = z_cores > 3


            #Uso de #IQR
            Q1 = numericos[columnas].quantile(0.25)
            Q3 = numericos[columnas].quantile(0.75)
            IQR = Q3 - Q1
            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR
            df[f'{columnas}_iqr_outlier'] = ~numericos[columnas].between(lim_inf, lim_sup)

        # Uso del Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)

        df['iso_outlier'] = iso_forest.fit_predict(numericos) == -1

        print(df)

        #Mostrar los datos que son outliers
        outliers = df[df.filter(like='_outlier').any(axis=1)]
        st.subheader("Filas detectadas como outliers")
        st.dataframe(outliers)

        columna = st.selectbox("Selecciona la variable a analizar", df.select_dtypes(include='number').columns)
        st.subheader(f"Boxplot para {columna}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[columna], ax=ax)
        st.pyplot(fig)



      
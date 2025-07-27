import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def cargar_archivos(archivos):
    for archivo in archivos:
        if archivo.name not in st.session_state.data_diccio:
            try:
                if archivo.name.endswith('.csv'):
                    df = pd.read_csv(archivo)
                else:
                    df = pd.read_excel(archivo)
                st.session_state.data_diccio[archivo.name] = df
                st.success(f"Archivo {archivo.name} cargado con éxito.")
            except Exception as e:
                st.error(f"No se pudo leer el archivo {archivo.name}: {e}")




def mostrar_archivos():
    if st.session_state.data_diccio:
        st.subheader("Archivos cargados")
        for fname in st.session_state.data_diccio:
            st.write(f"- {fname}")

        archivos_seleccionados = st.multiselect(
            "Selecciona uno o más archivos para analizar / combinar",
            options=list(st.session_state.data_diccio.keys())
        )

        if archivos_seleccionados:
            if len(archivos_seleccionados) == 1:
                df = st.session_state.data_diccio[archivos_seleccionados[0]]
                st.subheader(f"Vista previa de {archivos_seleccionados[0]}")
                #Se llama para la limpieza de datos
                df = limpieza_basica(df)
                st.dataframe(df.head())
                               
                
                st.session_state["df_actual"] = df

            else:
                st.subheader(f"Archivos combinados ({len(archivos_seleccionados)})")
                dfs = [st.session_state.data_diccio[f] for f in archivos_seleccionados]

                columnas_primero = set(dfs[0].columns)
                mismo_esquema = all(set(df.columns) == columnas_primero for df in dfs)

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



def limpieza_basica(df: pd.DataFrame) -> pd.DataFrame:
    st.write("### Limpieza")

    # Mostrar shape original
    st.write(f"Dimensiones originales: {df.shape}")

    #Eliminar filas con datos nulos
    df = df.dropna()
    # Eliminar duplicados
    
    
    df.duplicated()

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df.loc[:, col] = df[col].str.lower().str.strip()
   
    st.write(f"Dimensiones después de limpieza: {df.shape}")
    return df

def analisis_exploratorio(df: pd.DataFrame):
    st.subheader("Vista general del dataset")
    st.write(df.head(10).iloc[:, :20])

    st.subheader("Dimensiones")
    st.write(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

    #Mostrar los tipos de datos
    st.subheader("Tipos de datos")
    tipos_exactos = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de dato': df.dtypes.astype(str)
        })
    st.dataframe(tipos_exactos)
   
    #Mostar valores nulos
    # st.subheader("Valores nulos por columna")
    # st.write(df.isnull().sum())

    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())

    #Revisar
    # Separar numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns

    if len(columnas_numericas) > 0:
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

    if len(columnas_numericas) >= 2:
        st.subheader("Matriz de correlación")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap="YlGnBu", vmax=1, vmin=-1)
        st.pyplot(fig)

    if len(columnas_categoricas) > 0:
        st.subheader("Distribución de variables categóricas")
        variable_cat = st.selectbox("Selecciona variable categórica", columnas_categoricas)

        fig, ax = plt.subplots()
        df[variable_cat].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Distribución de {variable_cat}")
        st.pyplot(fig)


def outliers(df: pd.DataFrame):

    df = df.copy()


    df_numericos = df.select_dtypes(include=['int64','float64'])

    if df_numericos.empty:
        st.warning("No hay columnas numéricas en el archivo.")
    else:
        scalado = StandardScaler()

        scalado_data = scalado.fit_transform(df_numericos)

        st.subheader("Análisis de Outliers")
        contaminacion = st.slider("Proporción estimada de outliers (contamination)", 0.01, 0.2)


        iso_forest = IsolationForest(contamination=contaminacion, random_state=42)
        outliers = iso_forest.fit_predict(scalado_data)

        df['outlier_iso'] = outliers

        st.write("Cantidad de outliers detectados:")
        st.write(df['outlier_iso'].value_counts())


        # Mostrar tabla con outliers
        st.subheader("Outliers detectados")
        st.dataframe(df[df['outlier_iso'] == -1])

        # Selector de columna numérica para visualizar boxplot
        columna_seleccionada = st.selectbox("Selecciona la columna numérica para ver el boxplot", df_numericos.columns)

        # Graficar boxplot solo para la columna seleccionada
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df['outlier_iso'], y=df[columna_seleccionada])
        plt.title(f"Boxplot de {columna_seleccionada} según clasificación de outliers")
        plt.xlabel("Outlier (-1) / Normal (+1)")
        st.pyplot(plt)
        plt.clf()


      
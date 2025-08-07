import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.carga_archivos import *
from models.analisis_exploratotio import *
from models.analisis import *
from models.outliers import *
from models.clustering import *


st.set_page_config(page_title="Análisis Automático de Datos", layout="wide")



# Menú lateral para seleccionar la sección
st.sidebar.title("Dashboard")

# Menú lateral para seleccionar la sección
menu = st.sidebar.radio(
    "------------------------",
    (
        "1. Carga de archivos",
        "2. Análisis exploratorio",
        "3. outliers",
        "4. Clustering"
    )
)


# Variables global para almacenar datos cargados
#diccionario para guardar todos los archivos que se suben
if "data_diccio" not in st.session_state:
    st.session_state.data_diccio = {}

#Se guarda el cvs que se esta analizando en ese momento
if "df_actual" not in st.session_state:
    st.session_state.df_actual = None


# --- Sección 1. Carga de archivos ---
if menu == "1. Carga de archivos":
    st.title("Sistema Inteligente de Análisis Automatizado de Datos")
    st.header("Carga de archivos")

    uploaded_files = st.file_uploader(
        "Carga uno o varios archivos CSV o Excel",
        type=['csv', 'xlsx'],
        accept_multiple_files=True)
    
    if uploaded_files:
        cargar_archivos(uploaded_files)

    mostrar_archivos()

# --- Sección 2. Análisis exploratorio ---
if menu == "2. Análisis exploratorio":

    if st.session_state.df_actual is not None:
        df = st.session_state.df_actual
        analisis_exploratorio(df)
    else:
        st.warning("Primero debes cargar un archivo en la opción 1.")

# --- Sección 3. outliers ---
if menu == '3. outliers':
    st.header("outliers")
    if st.session_state.df_actual is not None:
        df = st.session_state.df_actual
        outliers(df)
    else:
        st.warning("Primero debes cargar un archivo en la opción 1.")

# --- Sección 4. Clustering ---
if menu == '4. Clustering':
    st.header("Clustering")

    if st.session_state.df_actual is not None:
        
        if 'outlier_general' in st.session_state.df_actual.columns:
            #Se guardan el df_filtrado los datos que en la columna outlier_general traen false
            df_filtrado = st.session_state.df_actual[~st.session_state.df_actual['outlier_general']].copy()      
            st.info(f"Se usarán {df_filtrado.shape[0]} registros después de filtrar outliers.")
        else:
            #Si no se paso por la opcion de outliers se utliliza el data frame original
            df_filtrado = df.copy()
        clustering(df_filtrado)     

    else:
        st.warning("Primero debes cargar un archivo en la opción 1.")




import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.analisis import *




st.set_page_config(page_title="Análisis Automático de Datos", layout="wide")

st.title("Sistema Inteligente de Análisis Automatizado de Datos")

# Menú lateral para seleccionar la sección
menu = st.sidebar.radio(
    "Navegación",
    (
        "1. Carga de archivos",
        "2. Análisis exploratorio",
        "3. Detección de valores atípicos",
        "",
        "",
        "",
        ""
    )
)


# Variable global para almacenar datos cargados
if "data_diccio" not in st.session_state:
    st.session_state.data_diccio = {}
if "df_actual" not in st.session_state:
    st.session_state.df_actual = None


# --- Sección 1: Carga de archivos ---
if menu == "1. Carga de archivos":
    st.header("1. Carga de archivos")

    uploaded_files = st.file_uploader(
        "Carga uno o varios archivos CSV o Excel",
        type=['csv', 'xlsx'],
        accept_multiple_files=True)
    
    if uploaded_files:
        cargar_archivos(uploaded_files)

    mostrar_archivos()
if menu == "2. Análisis exploratorio":
    st.header("2. Análisis Exploratorio de Datos")

    
    if st.session_state.df_actual is not None:
        df = st.session_state.df_actual
        analisis_exploratorio(df)
    else:
        st.warning("Primero debes cargar un archivo en la opción 1.")
if menu == '3. Detección de valores atípicos':
    st.header("3. Detección de valores atípicos")
    if st.session_state.df_actual is not None:
        df = st.session_state.df_actual
        outliers(df)
    else:
        st.warning("Primero debes cargar un archivo en la opción 1.")



# --- Aquí seguirían las demás secciones ---
# (Análisis estadístico, detección outliers, clustering, visualizaciones, resumen)
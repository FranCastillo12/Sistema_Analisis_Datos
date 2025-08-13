import pandas as pd
import streamlit as st


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
            "Selecciona uno o más archivos para analizar o combinar",
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
                    df_comb = limpieza_basica(df_comb)
                    
                    st.dataframe(df_comb.head())

                    st.session_state["df_actual"] = df_comb

                else:
                    st.warning("Los archivos seleccionados no tienen la misma estructura de columnas y no se pueden combinar.")
                    
            #Se descarga un csv con los datos limpios       
            st.download_button("Descargar datos limpios", df.to_csv(index=False), file_name="datos_limpios.csv")



                    #Funcion para limpiar los dataframe escogidos
def limpieza_basica(df: pd.DataFrame) -> pd.DataFrame:
    st.write("### Limpieza")

    # Mostrar shape original
    st.write(f"Dimensiones originales: {df.shape}")

    #Eliminar filas con datos nulos
    df = df.dropna()

    # Eliminar duplicados
    df = df.drop_duplicates()

    #Recorre las columnas que tienen datos object o string
    for col in df.select_dtypes(include=['object', 'string']).columns:
        #Pasa los datos de cada columna a minuscula
        df.loc[:, col] = df[col].str.lower().str.strip()
        
    #se imprimen los dataframen ya limpios
    st.write(f"Dimensiones después de limpieza: {df.shape}")
    return df

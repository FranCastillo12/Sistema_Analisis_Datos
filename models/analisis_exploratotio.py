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
    
    todas_columnas_categoricas = df.select_dtypes(include='object').columns.tolist()
    
    columnas_categoricas = [col for col in df.select_dtypes(include='object').columns 
                            if df[col].nunique() < 20]

    #Mostrar los tipos de datos
    st.subheader("Tipos de datos")
    tipos_exactos = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de dato': df.dtypes.astype(str)
        })
    
    st.dataframe(tipos_exactos)


    #Columnas categoricas
    st.subheader("Analisis de datos categoricos")
    #validacion para saber si se pueden mostrar el grafico
    if len(columnas_categoricas) > 0:
        
        #Se muestra grafico de distribucion de variavles

        st.subheader("Distribución de variables categóricas")
        variable_cat = st.selectbox("Selecciona una variable categórica", columnas_categoricas)
        
        
        #Crea una figura y un eje con un tamaño de 6x4 pulgadas.
        fig, ax = plt.subplots(figsize=(6, 4)) # Se utiliza para dibujar el gráfico
        
        #Grafico de barras
        sns.countplot(data=df, x=variable_cat, ax=ax, palette="pastel")

        ax.set_title(f"Distribución de {variable_cat}")
        
        #Etiqueta los ejes
        #Eje Y: Frecuencia (cantidad de veces que aparece cada categoría)
        #Eje X: Nombre de la variable categórica
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel(variable_cat)
        
        #Rota las etiquetas para que no se encimen si son largas.
        plt.xticks(rotation=45)  

        #Muestra la grafica
        st.pyplot(fig)
        
    # Mostrar todas las categóricas para eliminar (incluso si no son graficables)
    st.subheader("Eliminar columnas categóricas innecesarias")
    columnas_a_eliminar = st.multiselect(
    "Selecciona columnas que deseas eliminar del DataFrame",
    todas_columnas_categoricas)

    if st.button("Eliminar columnas seleccionadas"):
        
        #Se dice que columna se quiere eliminar y inplace true para decir que se 
        #haga el cambio directamente en el dataframe
        df.drop(columns=columnas_a_eliminar, inplace=True)
        st.success(f"Columnas eliminadas: {', '.join(columnas_a_eliminar)}")

        # Guardar el dataframe actualizado si estás usando session_state
        st.session_state.df_actual = df


    st.subheader("Analisis de datos numericos")

    st.dataframe(df.describe())



    #Si las columbas numericas son mayores a 0
    if len(columnas_numericas) > 0:

        #Se crean histogramas
        st.subheader("Histogramas")
        col1, col2 = st.columns(2)
        with col1:
            variable_num = st.selectbox("Selecciona variable numérica", columnas_numericas)
        with col2:
            #Controla la cantidad de rangos o barras que se muestran
            bins = st.slider("Cantidad de bins", max_value=100, value=10)

        fig, ax = plt.subplots(figsize=(6, 4))
        
        #kde=True: Agrega la curva de densidad sobre el histograma.
        sns.histplot(df[variable_num], bins=bins, kde=True, ax=ax)
        
        
        ax.set_title(f"Distribución de {variable_num}")
        st.pyplot(fig)

        st.subheader("Boxplot")
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



    st.subheader("Análisis Bivariado")

    #Se elige las columunas que se quieren relacionar
    col_x = st.selectbox("Variable 1", df.columns, key="varx")
    col_y = st.selectbox("Variable 2", df.columns, key="vary")
    
    
    if (col_x != col_y):
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Dos columnas númericas
        if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
            
            
            #Se utiliza un grafico de dispersion
            #sns.stripplot(data=df, x=col_x, y=col_y, ax=ax, jitter=0.2) 
            sns.set_style("whitegrid")
            
            sns.regplot(data=df, x=col_x, y=col_y, ax=ax,scatter_kws={'s': 25,'alpha': 0.6,'cmap': 'viridis'},x_jitter=0.5,y_jitter=10 ,line_kws={'color': 'black'},)
            #data=df(El DataFrame), x=col_x(variable para el eje x), y=col_y(variable para el eje y), ax=ax(donde se va dibujar el grafico), jitter=0.2(Agrega un ruido horizontal para que los puntos no se superpongan. )



            ax.set_title(f"Relación entre {col_x} y {col_y}")
            st.pyplot(fig)
            # Mostrar correlación
            corr = df[[col_x, col_y]].corr().iloc[0,1]
            
            
            #Muestra si existe una coerrelacion lineal entre las dos variables
            st.info(f"Coeficiente de correlación de Pearson: **{corr:.2f}**")

            # Interpretar la correlación
            if abs(corr) > 0.85:
                interpretacion = "prácticamente lineales"
            elif abs(corr) > 0.7:
                interpretacion = "fuertemente relacionadas"
            elif abs(corr) > 0.5:
                interpretacion = "moderadamente relacionadas"
            else:
                interpretacion = "débilmente relacionadas"

            # Mostrar interpretación
            st.info(f"**{col_x}** y **{col_y}** están {interpretacion} (r = {corr:.2f})")
    
        #Una columna categorica y numerica
        elif (
            pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_object_dtype(df[col_y]) or
            pd.api.types.is_object_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y])
        ):
            
            #Se utiliza grafico de cajas
            sns.boxplot(data=df, x=col_x, y=col_y, ax=ax)


            # data=df(El DataFrame), x=col_x(variable para el eje x), y=col_y(variable para el eje y), ax=ax(donde se va dibujar el grafico)
            ax.set_title(f"Distribución de {col_y} según {col_x}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Dos columnas categoricas
        elif pd.api.types.is_object_dtype(df[col_x]) and pd.api.types.is_object_dtype(df[col_y]):
            # Se utiliza tabla de frecuencia cruzada
            tabla = pd.crosstab(df[col_x], df[col_y])
            # Se utiliza un mapa de calor
            sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", ax=ax)
            sns.set_style("whitegrid")
            #tabla(Matriz de datos), annot=True(Muestra los valores dentro de cada celda del mapa), fmt="d"(Formato de los valores entero. ),cmap="Blues"(Paleta de colores) ,ax=ax(donde se va dibujar el grafico)
            ax.set_title(f"Frecuencia entre {col_x} y {col_y}")
            st.pyplot(fig)
            st.dataframe(tabla)

        else:
            st.warning("No se puede mostrar esta combinación de variables.")
    else:
        st.warning("Por favor selecciona dos variables distintas.")

        st.subheader("Analisis multivariadas")

    # st.subheader("Estadísticas generales")
    # st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    # st.dataframe(df.describe())

    # st.subheader("Vista general")
    # #Se muestra una vista general del dataframe
    # st.write(df.head(10).iloc[:, :20])











 

    


        





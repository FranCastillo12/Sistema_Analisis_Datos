import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


# Funcion para realizar el analisis del dataframe ya limpio
def analisis_exploratorio(df: pd.DataFrame):

    st.title("Exploración y Visualización")
    
    # Separar numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns

    todas_columnas_categoricas = df.select_dtypes(include="object").columns.tolist()

    columnas_categoricas = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() < 20
    ]

    with st.expander("Tipos de datos"):

        # Mostrar los tipos de datos
        st.subheader("Tipos de datos")
        tipos_exactos = pd.DataFrame(
            {"Columna": df.columns, "Tipo de dato": df.dtypes.astype(str)}
        )

        st.dataframe(tipos_exactos)

    with st.expander("Analisis de datos categoricos"):

        #  validacion para saber si se pueden mostrar el grafico
        if len(columnas_categoricas) > 0:

            # Se muestra grafico de distribucion de variavles

            st.subheader("Distribución de variables categóricas")
            variable_cat = st.selectbox(
                "Selecciona una variable categórica", columnas_categoricas
            )

            # Crea una figura y un eje con un tamaño de 6x4 pulgadas.
            fig, ax = plt.subplots(figsize=(6, 4))  # Se utiliza para dibujar el gráfico

            # Grafico de barras
            sns.countplot(data=df, x=variable_cat, ax=ax, palette="pastel")

            ax.set_title(f"Distribución de {variable_cat}")

            # Etiqueta los ejes
            # Eje Y: Frecuencia (cantidad de veces que aparece cada categoría)
            # Eje X: Nombre de la variable categórica
            ax.set_ylabel("Frecuencia")
            ax.set_xlabel(variable_cat)

            # Rota las etiquetas para que no se encimen si son largas.
            plt.xticks(rotation=45)

            # Muestra la grafica
            st.pyplot(fig)

        # Mostrar todas las categóricas para eliminar (incluso si no son graficables)
        st.subheader("Eliminar columnas categóricas innecesarias")
        columnas_a_eliminar = st.multiselect(
            "Selecciona columnas que deseas eliminar del DataFrame",
            todas_columnas_categoricas,
        )

        if st.button("Eliminar columnas seleccionadas"):

            # Se dice que columna se quiere eliminar y inplace true para decir que se
            # haga el cambio directamente en el dataframe
            df.drop(columns=columnas_a_eliminar, inplace=True)
            st.success(f"Columnas eliminadas: {', '.join(columnas_a_eliminar)}")

            # Guardar el dataframe actualizado si estás usando session_state
            st.session_state.df_actual = df
            
            
            
    with st.expander("Analisis de datos numericos"):
        
        st.subheader("Analisis de datos numericos")

        st.dataframe(df.describe())

        #Insight del analsiis estadistico
        st.markdown("**Insight automático:**")
        st.markdown(f"- El promedio de valores numéricos más alto está en **{df[columnas_numericas].mean().idxmax()}**.")
        st.markdown(f"- La variable con mayor dispersión (desviación estándar) es **{df[columnas_numericas].std().idxmax()}**.")
        
        
        # Si las columbas numericas son mayores a 0
        if len(columnas_numericas) > 0:

            # Se crean histogramas
            st.subheader("Histogramas")
            col1, col2 = st.columns(2)
            with col1:
                variable_num = st.selectbox(
                    "Selecciona variable numérica", columnas_numericas)
            with col2:
                # Controla la cantidad de rangos o barras que se muestran
                bins = st.slider("Cantidad de bins", max_value=100, value=10)

            fig, ax = plt.subplots(figsize=(6, 4))

            # kde=True: Agrega la curva de densidad sobre el histograma.
            sns.histplot(df[variable_num], bins=bins, kde=True, ax=ax)

            ax.set_title(f"Distribución de {variable_num}")
            st.pyplot(fig)

            st.subheader("Boxplot")
            var_box = st.selectbox(
                "Selecciona variable numérica", columnas_numericas, key="box")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[var_box], ax=ax)
            ax.set_title(f"Boxplot de {var_box}")
            st.pyplot(fig)

    
    with st.expander("Análisis Bivariado"):
        st.subheader("Análisis Bivariado")

        # Se elige las columunas que se quieren relacionar
        col_x = st.selectbox("Variable 1", df.columns, key="varx")
        col_y = st.selectbox("Variable 2", df.columns, key="vary")

        if col_x != col_y:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Dos columnas númericas
            if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(
                df[col_y]):

                # Se utiliza un grafico de dispersion


                sns.regplot(
                    data=df, #dataframe que contiene los datos
                    x=col_x, #varibales para graficas
                    y=col_y, #varibales para graficas
                    ax=ax, #eje donde se dibuja el grafico
                    scatter_kws={"s": 25, "alpha": 0.6, "cmap": "viridis"}, #personaliza los puntos: tamaño, transparencia y color.
                    x_jitter=0.5, #Ruido para evitar que los puntosse superpongan
                    y_jitter=10, #Ruido para evitar que los puntosse superpongan
                    line_kws={"color": "black"},) # color de la línea de tendencia
                    
                ax.set_title(f"Relación entre {col_x} y {col_y}")
                st.pyplot(fig)
                # Mostrar correlación
                corr = df[[col_x, col_y]].corr().iloc[0, 1] # Calcula el coeficiente de correlación de Pearson entre las dos variables.

                # Muestra si existe una coerrelacion lineal entre las dos variables
                st.info(f"Coeficiente de correlación de Pearson: **{corr:.2f}**")

                # Interpretar la correlación
                
                if abs(corr) > 0.85: #  prácticamente lineales
                    interpretacion = "prácticamente lineales"
                elif abs(corr) > 0.7: #fuertemente relacionadas
                    interpretacion = "fuertemente relacionadas"
                elif abs(corr) > 0.5: #fuertemente relacionadas
                    interpretacion = "moderadamente relacionadas"
                else: #débilmente relacionadas
                    interpretacion = "débilmente relacionadas"

                # Mostrar interpretación
                st.info(f"**{col_x}** y **{col_y}** están {interpretacion} (r = {corr:.2f})" )
                
                # Insight automático: relación detectada
                st.markdown("**Insight automático:**")
                if abs(corr) > 0.7:
                    st.markdown(f"- Existe una relación significativa entre **{col_x}** y **{col_y}** (r = {corr:.2f}). Esto podría indicar dependencia o influencia directa.")
                else:
                    st.markdown(f"- La relación entre **{col_x}** y **{col_y}** es débil (r = {corr:.2f}). Puede no ser relevante para predicción.")

            # Una columna categorica y numerica
            elif (
                pd.api.types.is_numeric_dtype(df[col_x])
                and pd.api.types.is_object_dtype(df[col_y])
                or pd.api.types.is_object_dtype(df[col_x])
                and pd.api.types.is_numeric_dtype(df[col_y])):

                # Se utiliza grafico de cajas
                sns.boxplot(data=df, x=col_x, y=col_y, ax=ax)

                # data=df(El DataFrame), x=col_x(variable para el eje x), y=col_y(variable para el eje y), ax=ax(donde se va dibujar el grafico)
                ax.set_title(f"Distribución de {col_y} según {col_x}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # Dos columnas categoricas
            elif pd.api.types.is_object_dtype(df[col_x]) and pd.api.types.is_object_dtype(
                df[col_y] ):
                
                
                # Se utiliza tabla de frecuencia cruzada
                tabla = pd.crosstab(df[col_x], df[col_y])
                # Se utiliza un mapa de calor
                sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", ax=ax)
                sns.set_style("whitegrid")
                # tabla(Matriz de datos), annot=True(Muestra los valores dentro de cada celda del mapa), fmt="d"(Formato de los valores entero. ),cmap="Blues"(Paleta de colores) ,ax=ax(donde se va dibujar el grafico)
                ax.set_title(f"Frecuencia entre {col_x} y {col_y}")
                st.pyplot(fig)
                st.dataframe(tabla)

            else:
                st.warning("No se puede mostrar esta combinación de variables.")
        else:
            st.warning("Por favor selecciona dos variables distintas.")
    with st.expander("Analisis multivariadas"):
        
        st.subheader("Analisis multivariadas")

        if len(columnas_numericas) > 2:
            
            # Seleccionar variables
            seleccion_multi = st.multiselect(
                "Selecciona variables numéricas para análisis multivariado (máximo 5)",
                columnas_numericas,
                default=columnas_numericas[:3],
                max_selections=5)
            
            if len(seleccion_multi) >= 2:
                fig = sns.pairplot(df[seleccion_multi], diag_kind="kde", corner=True)
                st.pyplot(fig)

                st.subheader("Matriz de correlación")
                corr = df[seleccion_multi].corr()

                fig2, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Matriz de correlación entre variables seleccionadas")
                st.pyplot(fig2)
                
                # Insight automático: correlación multivariada
                st.markdown("**Insight automático:**")
                corr_matrix = df[seleccion_multi].corr()
                
                max_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
                if not max_corr.empty:
                    top_pair = max_corr.index[0]
                    st.markdown(f"- Las variables **{top_pair[0]}** y **{top_pair[1]}** tienen la correlación más alta entre las seleccionadas (r = {max_corr.iloc[0]:.2f}).")

            else:
                st.warning("Se necesitan seleccionar al menos dos variables.")
                # Matriz de correlación

        else:
            st.warning("Se necesitan al menos dos variables numéricas para análisis multivariado.")
            
    st.markdown("**Resumen general del análisis exploratorio:**")
    st.markdown(f"- Se detectaron {len(columnas_numericas)} variables numéricas y {len(columnas_categoricas)} categóricas con menos de 20 categorías.")
    st.markdown(f"- El DataFrame tiene {df.shape[0]} registros y {df.shape[1]} columnas después de limpieza.")
            
            
    # st.subheader("Estadísticas generales")
    # st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    # st.dataframe(df.describe())

    # st.subheader("Vista general")
    # #Se muestra una vista general del dataframe
    # st.write(df.head(10).iloc[:, :20])

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt



#Funcion para realizar outliers
def outliers(df: pd.DataFrame):
    
    
    st.header("Detección de Outliers")
    with st.expander("outliers"):
        
        

        #Se crea una copia del dataframe que se esta analizando
        df = df.copy()

        #Se guardan las columnas que tienen datos numericos
        numericos = df.select_dtypes(include=[np.number])

        if numericos.empty:
            st.warning("No hay columnas numéricas en el archivo.")
            return

        # Z-Score
        z_outliers = (np.abs(stats.zscore(numericos)) > 3).any(axis=1)
        #Se le agrega colunma al df
        df['outlier_zscore'] = z_outliers

        # IQR
        iqr_outliers = pd.Series(False, index=df.index)
        for col in numericos.columns:
            Q1 = numericos[col].quantile(0.25) #datos son menores o iguales a este valor
            Q3 = numericos[col].quantile(0.75)  #datos son menores o iguales a este valor.
            #Rango intercuatil
            IQR = Q3 - Q1  # dispersión de ese 50% central.
            
            mascara = ~numericos[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR) # se difinen los limites para detectar los outliers
            
            iqr_outliers = iqr_outliers | mascara # se agrega true donde hay outliers
        df['outlier_iqr'] = iqr_outliers

        # Isolation Forest
        modelo = IsolationForest(contamination=0.1, random_state=42)
        df['outlier_isolation'] = modelo.fit_predict(numericos) == -1

        # 
        df['outlier_general'] = df[['outlier_zscore', 'outlier_iqr', 'outlier_isolation']].any(axis=1)

        #Guardar el df en el df original
        st.session_state.df_actual = df


        outliers_detectados = df[df['outlier_general'] == True]
        cantidad = len(outliers_detectados)
        st.success(f"Se detectaron {cantidad} filas con al menos un valor atípico.")
        
        # Insight automático
        st.markdown("**Insight automático:**")

        if cantidad == 0:
            st.markdown("- No se detectaron valores atípicos significativos en las variables numéricas. Esto sugiere una distribución relativamente estable.")
        else:
            porcentaje = cantidad / len(df) * 100
            st.markdown(f"- Se detectaron **{cantidad}** filas con outliers, lo que representa aproximadamente **{porcentaje:.2f}%** del total.")
    
            # Variables más afectadas
            columnas_afectadas = df.loc[df['outlier_general']].select_dtypes(include='number').apply(
                lambda col: col.isna().sum() + (col == 0).sum() + col.duplicated().sum(), axis=0
                ).sort_values(ascending=False)

            top_vars = columnas_afectadas.head(3).index.tolist()
            if top_vars:
                st.markdown(f"- Las variables más frecuentemente asociadas a outliers son: **{', '.join(top_vars)}**.")
                
        st.dataframe(outliers_detectados)


        columna = st.selectbox("Selecciona la variable a analizar", df.select_dtypes(include='number').columns)
        st.subheader(f"Boxplot para {columna}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[columna], ax=ax)
        st.pyplot(fig)

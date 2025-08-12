import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import plot_tree



def prediccion_datos(df: pd.DataFrame):
    st.header("Predicción de datos")
    

    target = st.selectbox("Seleccione la columna que desea predecir", df.columns)  
    # Lista de columnas para X
    columnas_X = [col for col in df.columns if col != target]
    variables_X = st.multiselect("Seleccione las columnas para usar como variables independientes (X)", columnas_X, default=columnas_X)

    if target and variables_X:
        X = df[variables_X]
        y = df[target]
        
        
        
    
            
        # Codificar target si es categórico
        le_target = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
    

        problem_type = "clasificacion" if le_target is not None else "regresion"
        st.write(f"Tipo de problema detectado: {problem_type}")

        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Elegir modelo inicial
        if problem_type == "regresion":
            modelo_sel = st.selectbox("Elige el modelo para regresión:", ["LinearRegression", "RandomForestRegressor"])
            if modelo_sel == "LinearRegression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            modelo_sel = st.selectbox("Elige el modelo para clasificación:", ["LogisticRegression", "RandomForestClassifier", "SVC"])
            if modelo_sel == "LogisticRegression":
                model = LogisticRegression(max_iter=300)
            elif modelo_sel == "RandomForestClassifier":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = SVC(probability=True)

        # Entrenar modelo inicial
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Mostrar métricas y gráficos iniciales
        if problem_type == "regresion":
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Error cuadrático medio (MSE): {mse:.2f}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Valores reales")
            ax.set_ylabel("Valores predichos")
            ax.set_title("Real vs Predicho")
            st.pyplot(fig)
            
            if modelo_sel == "RandomForestRegressor":
                importances = model.feature_importances_
                fig2, ax2 = plt.subplots()
                sns.barplot(x=importances, y=X.columns, ax=ax2)
                ax2.set_title("Importancia de variables")
                st.pyplot(fig2)
        else:
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Precisión (accuracy): {acc:.2f}")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=ax)
            st.pyplot(fig)
            
            if modelo_sel == "RandomForestClassifier":
                importances = model.feature_importances_
                fig2, ax2 = plt.subplots()
                sns.barplot(x=importances, y=X.columns, ax=ax2)
                ax2.set_title("Importancia de variables")
                st.pyplot(fig2) 

        # --- NUEVA PARTE: selección de variables importantes y reentrenamiento ---

        st.subheader("Entrenar modelo con variables más influyentes")

        # Entrenar un RandomForest para calcular importancias (clasificación o regresión)
        if problem_type == "regresion":
            rf_temp = RandomForestRegressor(random_state=42)
        else:
            rf_temp = RandomForestClassifier(random_state=42)

        rf_temp.fit(X_train, y_train)
        importancias = pd.Series(rf_temp.feature_importances_, index=X.columns).sort_values(ascending=False)

        # Mostrar gráfico de importancias
        fig, ax = plt.subplots()
        importancias.plot(kind='barh', ax=ax)
        ax.invert_yaxis()
        ax.set_title("Importancia de variables (RandomForest)")
        st.pyplot(fig)

        # Seleccionar variables con importancia > 0.01
        columnas_influyentes = importancias[importancias > 0.01].index.tolist()

        if len(columnas_influyentes) == 0:
            st.warning("Ninguna variable supera el umbral de importancia 0.01, se usarán todas las variables.")
            columnas_influyentes = X.columns.tolist()
        else:
            st.info(f"Variables seleccionadas: {columnas_influyentes}")

        # Re-entrenar con variables seleccionadas
        X_train_sel = X_train[columnas_influyentes]
        X_test_sel = X_test[columnas_influyentes]

        if problem_type == "regresion":
            modelo_final = RandomForestRegressor(random_state=42)
            modelo_final.fit(X_train_sel, y_train)
            pred = modelo_final.predict(X_test_sel)

            st.write("**R2 Score:**", r2_score(y_test, pred))
            st.write("**MAE:**", mean_absolute_error(y_test, pred))

            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=pred, ax=ax)
            ax.set_xlabel("Valores Reales")
            ax.set_ylabel("Predicciones")
            ax.set_title("Real vs Predicho (Regresión)")
            st.pyplot(fig)

            errores = y_test - pred
            fig, ax = plt.subplots()
            sns.histplot(errores, kde=True, ax=ax)
            ax.set_title("Distribución de Errores")
            st.pyplot(fig)

            fig, ax = plt.subplots()
            importancias_final = pd.Series(modelo_final.feature_importances_, index=columnas_influyentes)
            importancias_final.sort_values().plot(kind='barh', figsize=(8,6), ax=ax)
            ax.set_title("Importancia de Variables (Modelo Final)")
            st.pyplot(fig)

        else:
            modelo_final = RandomForestClassifier(random_state=42)
            modelo_final.fit(X_train_sel, y_train)
            pred = modelo_final.predict(X_test_sel)

            st.write("**Accuracy:**", accuracy_score(y_test, pred))

            fig, ax = plt.subplots()
            cm_final = confusion_matrix(y_test, pred)
            disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final)
            disp_final.plot(cmap="Blues", ax=ax)
            st.pyplot(fig)

            fig, ax = plt.subplots()
            pd.Series(pred).value_counts().plot(kind='bar', alpha=0.7, label="Predicho", ax=ax)
            pd.Series(y_test).value_counts().plot(kind='bar', alpha=0.7, label="Real", ax=ax)
            plt.legend()
            ax.set_title("Comparación de Clases Reales vs Predichas")
            st.pyplot(fig)

            fig, ax = plt.subplots()
            importancias_final = pd.Series(modelo_final.feature_importances_, index=columnas_influyentes)
            importancias_final.sort_values().plot(kind='barh', figsize=(8,6), ax=ax)
            ax.set_title("Importancia de Variables (Modelo Final)")
            st.pyplot(fig)

            # Opcional: mostrar árbol de decisión del primer estimador
            if st.checkbox("Mostrar árbol de decisión del primer estimador"):
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(modelo_final.estimators_[0], feature_names=columnas_influyentes, filled=True, rounded=True, ax=ax)
                st.pyplot(fig)

        # if np.issubdtype(y.dtype, np.number):
        #     modelo_temp = RandomForestRegressor(random_state=42)
        # else:
        #     modelo_temp = RandomForestClassifier(random_state=42)

        # modelo_temp.fit(X_train, y_train)
        # importancias = pd.Series(modelo_temp.feature_importances_, index=X.columns).sort_values(ascending=False)

        # # Mostrar gráfico de importancia
        # st.subheader("Importancia de Variables")
        # fig, ax = plt.subplots()
        # importancias.plot(kind='barh', ax=ax)
        # st.pyplot(fig)

        # # Seleccionar solo las más influyentes (ej. importancia > 0.01)
        # columnas_influyentes = importancias[importancias > 0.01].index
        # X_train = X_train[columnas_influyentes]
        # X_test = X_test[columnas_influyentes]

        # st.info(f"Entrenando modelo solo con variables influyentes: {list(columnas_influyentes)}")
        
        # # Si es regresión
        # if np.issubdtype(y.dtype, np.number):
        #     modelo = RandomForestRegressor(random_state=42)
        #     modelo.fit(X_train, y_train)
        #     pred = modelo.predict(X_test)

        #     st.write("**R2 Score:**", r2_score(y_test, pred))
        #     st.write("**MAE:**", mean_absolute_error(y_test, pred))

        #     # Gráfico Real vs Predicho
        #     fig, ax = plt.subplots()
        #     sns.scatterplot(x=y_test, y=pred, ax=ax)
        #     ax.set_xlabel("Valores Reales")
        #     ax.set_ylabel("Predicciones")
        #     ax.set_title("Real vs Predicho (Regresión)")
        #     st.pyplot(fig)

        #     # Histograma de errores
        #     errores = y_test - pred
        #     fig, ax = plt.subplots()
        #     sns.histplot(errores, kde=True, ax=ax)
        #     ax.set_title("Distribución de Errores")
        #     st.pyplot(fig)

        #     # Feature importance
        #     fig, ax = plt.subplots()
        #     importancias = pd.Series(modelo.feature_importances_, index=X.columns)
        #     importancias.sort_values().plot(kind='barh', figsize=(8,6), ax=ax)
        #     ax.set_title("Importancia de Variables")
        #     st.pyplot(fig)

        # # Si es clasificación
        # else:
        #     modelo = RandomForestClassifier(random_state=42)
        #     modelo.fit(X_train, y_train)
        #     pred = modelo.predict(X_test)

        #     st.write("**Accuracy:**", accuracy_score(y_test, pred))

        #     # Matriz de confusión
        #     fig, ax = plt.subplots()
        #     cm = confusion_matrix(y_test, pred)
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #     disp.plot(cmap="Blues", ax=ax)
        #     st.pyplot(fig)

        #     # Comparación de clases
        #     fig, ax = plt.subplots()
        #     pd.Series(pred).value_counts().plot(kind='bar', alpha=0.7, label="Predicho", ax=ax)
        #     y_test.value_counts().plot(kind='bar', alpha=0.7, label="Real", ax=ax)
        #     plt.legend()
        #     ax.set_title("Comparación de Clases Reales vs Predichas")
        #     st.pyplot(fig)

        #     # Feature importance
        #     fig, ax = plt.subplots()
        #     importancias = pd.Series(modelo.feature_importances_, index=X.columns)
        #     importancias.sort_values().plot(kind='barh', figsize=(8,6), ax=ax)
        #     ax.set_title("Importancia de Variables")
        #     st.pyplot(fig)

        #     # Árbol de decisión
        #     fig, ax = plt.subplots(figsize=(20,10))
        #     plot_tree(modelo.estimators_[0], feature_names=X.columns, filled=True, rounded=True, ax=ax)
        #     st.pyplot(fig)

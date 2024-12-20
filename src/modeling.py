from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np


########################################################################################################
########################################################################################################

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model="Random Forest", seed=42):
    """
    Entrena y evalúa un modelo de clasificación utilizando uno de varios algoritmos predefinidos.

    Parámetros:
    - X_train (DataFrame o array): Datos de entrada para el entrenamiento.
    - X_test (DataFrame o array): Datos de entrada para la evaluación.
    - y_train (Series o array): Etiquetas correspondientes para el entrenamiento.
    - y_test (Series o array): Etiquetas correspondientes para la evaluación.
    - model (str, opcional): Nombre del modelo a utilizar. Por defecto, "Random Forest".
      Opciones disponibles:
        - "Logistic Regression"
        - "Random Forest"
        - "Support Vector Machine"
        - "Decision Tree"
        - "K-Nearest Neighbors"
        - "Naive Bayes"
        - "Gradient Boosting"
    - seed (int, opcional): Semilla para garantizar reproducibilidad. Por defecto es 42.

    Proceso:
    1. Define un diccionario `models` con instancias preconfiguradas de algoritmos de clasificación.
    2. Selecciona el modelo especificado en el parámetro `model`.
    3. Entrena el modelo seleccionado utilizando los datos de entrenamiento.
    4. Evalúa el modelo en los datos de prueba y calcula la exactitud.

    Retorna:
    - model (objeto): Modelo entrenado.
    - accuracy (float): Exactitud del modelo en el conjunto de prueba.

    Ejemplo de uso:
    >>> model, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, model="Logistic Regression")
    >>> print(f"Exactitud del modelo: {accuracy:.2f}")
    """
    random_state = seed  # Garantiza reproducibilidad

    # Diccionario de modelos predefinidos
    models = {
        "Logistic Regression": LogisticRegression(max_iter=100000, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }

    # Verificar si el modelo seleccionado está en el diccionario
    if model not in models:
        raise ValueError(f"El modelo '{model}' no está soportado. Las opciones disponibles son: {list(models.keys())}")

    # Seleccionar y entrenar el modelo
    model_instance = models[model]
    model_instance.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model_instance.predict(X_test)
    y_proba = model_instance.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = model_instance.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print("Modelo utilizado: ", model)
    print(f"Exactitud del modelo: {accuracy:.2f}")

    return model_instance, accuracy, precision, recall, f1, auc

########################################################################################################
########################################################################################################

def evaluate_models(X_train, X_test, y_train, y_test, test_size=0.2, random_state=42):
    """
    Evalúa múltiples modelos de clasificación en un conjunto de datos dado.
    Imprime la evaluacion. 
    
    Parámetros:
        X (DataFrame): Características (features).
        y (Series o array): Variable objetivo (target).
        test_size (float): Tamaño del conjunto de prueba (default: 0.2).
        random_state (int): Semilla para reproducibilidad (default: 42).
    
    Retorna:
        DataFrame con las métricas de evaluación para cada modelo.
    """
    # Dividir los datos
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Modelos a evaluar
    models = {
        "Logistic Regression": LogisticRegression(max_iter=100000, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }
    
    # Almacenar resultados
    results = []

    i=0
    for name, model in models.items():

        # Entrenar el modelo
        if i==0:
            from sklearn.preprocessing import StandardScaler
            # Escalar datos
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            
            #X_test_scaled
        
        else:
            model.fit(X_train, y_train)
        
        # Predicciones
        if i==0:
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            i=i+1
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            
        # Guardar resultados
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc
        })
    
    # Convertir a DataFrame
    df_model_eval = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    best_model_name = df_model_eval.iloc[0]["Model"]
    print(df_model_eval)
    print("Best performance model in Accuracy is: ", best_model_name)
    return df_model_eval, best_model_name

########################################################################################################
########################################################################################################




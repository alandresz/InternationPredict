from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

import os
import pickle
import pandas as pd
from datetime import datetime

##################################################################################################

def full_metrics_eval(model, X_train, y_train, X_test, y_test, cv_split=10):
    """
    Evalúa el desempeño de un modelo de clasificación utilizando métricas clave como validación cruzada, matriz de confusión 
    y reporte de clasificación. 

    Args:
        model (sklearn.base.BaseEstimator): El modelo de clasificación a evaluar.
        X_train (pd.DataFrame o np.ndarray): Conjunto de características de entrenamiento.
        y_train (pd.Series o np.ndarray): Etiquetas de entrenamiento.
        X_test (pd.DataFrame o np.ndarray): Conjunto de características de prueba.
        y_test (pd.Series o np.ndarray): Etiquetas de prueba.
        cv_split (int, opcional): Número de divisiones para la validación cruzada. Por defecto es 10.

    Returns:
        tuple: 
            - cv_score (np.ndarray): Puntajes de la validación cruzada.
            - y_pred (np.ndarray): Predicciones realizadas por el modelo en el conjunto de prueba.
            - cm (np.ndarray): Matriz de confusión.
            - cr (str): Reporte de clasificación en formato de texto.
    
    Prints:
        - Promedio de los puntajes de validación cruzada.
        - Matriz de confusión.
        - Reporte de clasificación.
    """
    # Realizar validación cruzada
    cv_score = cross_val_score(model, X_train, y_train, cv=cv_split)
    print(f"Promedio de los puntajes de validación cruzada: {cv_score.mean():.2f}")

    # Generar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(cm)

    # Imprimir matriz de confusión porcentual
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print("Matriz de confusión porcentual:")
    print(np.round(cm_percent, 2))

    # Calcular y mostrar el reporte de clasificación
    cr = classification_report(y_test, y_pred)
    print("Reporte de Clasificación:")
    print(cr)

    return cv_score, y_pred, cm, cr

####################################################################################################
####################################################################################################


########################################################################################################
########################################################################################################

def save_model_with_metadata(
    model, model_name, metrics, cv_score, confusion_matrix, classification_report, config, 
    model_dir="models", log_file="model_log.csv"
):
    """
    Guarda el modelo junto con metadatos relevantes en un único archivo .pkl.
    También registra sus detalles en un archivo CSV.

    Args:
        model (sklearn.BaseEstimator): Modelo entrenado.
        model_name (str): Nombre del modelo (ejemplo: "Random Forest").
        metrics (dict): Diccionario con las métricas del modelo (accuracy, precision, recall, f1-score).
        cv_score (float): Promedio de cross-validation.
        confusion_matrix (array): Matriz de confusión generada por sklearn.
        classification_report (str): Reporte de clasificación generado por sklearn.
        config (object): Configuración utilizada durante el entrenamiento, como un objeto de clase Config.
        model_dir (str): Directorio donde se guardará el archivo. Por defecto, "models".
        log_file (str): Archivo CSV donde se registrarán los modelos. Por defecto, "model_log.csv".

    Returns:
        str: Ruta del archivo guardado.
    """
    # Crear nombre del archivo del modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{metrics['accuracy']:.2f}_{metrics['recall']:.2f}_{timestamp}.pkl"
    filepath = os.path.join(model_dir, filename)

    # Crear el directorio de modelos si no existe
    os.makedirs(model_dir, exist_ok=True)

    # Crear un diccionario con el modelo y los metadatos
    data_to_save = {
        "model": model,
        "metrics": metrics,
        "cv_score": cv_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "config": config
    }

    # Guardar en un archivo pickle
    with open(filepath, "wb") as file:
        pickle.dump(data_to_save, file)
    print(f"Modelo y metadatos guardados en: {filepath}")

    # Registro en el archivo CSV
    log_entry = {
        "Nombre del archivo del modelo": filename,
        "Fecha y hora": timestamp,
        "Accuracy": metrics['accuracy'],
        "Precision": metrics['precision'],
        "Recall": metrics['recall'],
        "F1-Score": metrics['f1'],
        "Cross-Validation": cv_score.mean() if hasattr(cv_score, 'mean') else cv_score
    }

    if os.path.exists(log_file):
        # Leer archivo existente
        log_df = pd.read_csv(log_file)
    else:
        # Crear un nuevo DataFrame si no existe el archivo
        log_df = pd.DataFrame(columns=log_entry.keys())

    # Agregar nuevo registro
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)

    # Guardar el DataFrame actualizado
    log_df.to_csv(log_file, index=False)
    print(f"Registro actualizado en: {log_file}")

    return filepath


#############################################################################################
#############################################################################################

def load_model_with_metadata(filepath):
    """
    Carga un modelo guardado con sus metadatos desde un archivo .pkl.
    Valida que el archivo contiene las claves esperadas.
    
    Args:
        filepath (str): Ruta al archivo .pkl que contiene el modelo y sus metadatos.
    
    Returns:
        dict: Diccionario con el modelo y los metadatos asociados.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
    
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    # Validar que contiene las claves necesarias
    expected_keys = {"model", "metrics", "cv_score", "confusion_matrix", "classification_report", "config"}
    if not expected_keys.issubset(data.keys()):
        raise ValueError(f"El archivo no contiene todas las claves esperadas. Claves encontradas: {data.keys()}")
    
    print("Modelo y metadatos cargados exitosamente.")
    return data
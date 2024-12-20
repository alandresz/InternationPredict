Autor: Alan Dreszman, Ing. 
Fecha: 20-12-2024


# README - Proyecto de Modelado Predictivo


Observacion inicial:
-Para ver una descripción visual del flujo del proyecto si necesidad de abrir la notebook principal "Informe.ipynb" se puede consultar "Informe.html" que contiene una webpage interactiva con los resultados de una ejecución satisfactoria.

-No se recomienda consultar "Informe.pdf" ya que recorta algunas partes que pueden ser de interés.


## Descripción General

Este proyecto implementa un modelo predictivo para clasificar si un paciente será internado o no al llegar a la guardia de un hospital. Se procesan datos de pacientes, episodios médicos, estudios complementarios y signos vitales, generando un conjunto de datos enriquecido para el entrenamiento y evaluación de varios modelos de machine learning.



### Características Principales
- **Procesamiento de datos**: Limpieza y transformación automática de los datos brutos.
- **Modelos evaluados**: Logistic Regression, Random Forest, SVM, Decision Tree, K-Nearest Neighbors, Naive Bayes y Gradient Boosting.
- **Evaluación de métricas**: Cross-validation, matriz de confusión, reporte de clasificación.
- **Almacenamiento**: Guardado de modelos entrenados junto con metadatos y configuraciones para futura reutilización.


---

## Estructura del Proyecto

### Archivos y Directorios Principales
- **Informe.ipynb**: Es el archivo que muestra y describe el flujo de trabajo desde el inicio al final
- **Debug.ipynb**: Utilizado para probar modularmente el desarrollo. Parecido a **Informe** pero menos detallado
- **main_hospital_classifier.py**: Corre todo el proceso en un único script.
- **src/preprocessing.py**: Contiene funciones para cargar, limpiar y preparar los datos.
- **src/modeling.py**: Incluye funciones para entrenar modelos, así como guardar resultados.
- **src/evaluation.py**: Incluye funciones para evaluar modelos, así como guardar resultados.
- **data/**: Directorio donde deben ubicarse los archivos CSV originales.
- **models/**: Directorio donde se guardan los modelos entrenados junto con sus metadatos.
- **notebooks/**: Contiene notebooks Jupyter utilizadas como Sandbox. (no icluido en esta entrega)

Se debe respetar la estructura del sistema de archivos ya que tanto Informe.ipynb, Debug.ipynb y main_hospital_classifier.py referencian archivos en los directorios mencionados.

-Se puede ejecutar directamente como "python main_hospital_classifier.py"

OBS: Idealmente se debería ejecutar la solucion en un ambiente virtual o compartimentalizado

---

## Requisitos del Sistema
- **Python**: 3.11 o superior.
- **Librerías principales**:
  - pandas
  - numpy
  - scikit-learn
  - ydata-profiling
  - pickle
  - chardet

Instalar todas las dependencias ejecutando:
```
"pip install -r requirements.txt" o "conda install -c conda-forge requirements.txt"
```

---

## Guía de Uso

### 1. Configuración Inicial
En el archivo `config.py` si se desea cambiar:
- Ruta a los datos.
- Características consideradas para el modelo.

Los hiperparámetros del modelo por ahora se ajustan directamente sobre el final del archivo "informe.ipynb" (notebook) o "main_hospital_clasifier.py" (script)



### 2. Ejecución del Proyecto

#### Preprocesamiento de Datos
1. Importar y cargar los datos:
    ```python
    from preprocessing import load_and_read_data, default_clean

    # Cargar datos
    archivos = ["Pacientes.csv", "Episodios_Diagnosticos.csv", "Estudios_Complementarios.csv", "Signos_Vitales.csv"]
    path = "data/"
    data = load_and_read_data(archivos, path)

    ```
2. Limpiar y preparar los datos:
    ```python
    from preprocessing import create_final_dataframe
    df_final = create_final_dataframe(*data)
    ```


#### Entrenamiento y Evaluación
1. Dividir los datos en entrenamiento y prueba:
    ```python
    from preprocessing import prepare_train_test

    
    X_train, X_test, y_train, y_test = prepare_test_train(df_final, test_size=config.test_size, random_state=config.seed)
    ```


2. Entrenar y evaluar un modelo:
    ```python
    from modeling import train_and_evaluate_model

    model, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, model=config.model)
    ```

#### Guardado del Modelo
1. Guardar el modelo entrenado junto con los metadatos:
    ```python
    from modeling import save_model_with_metadata

    metrics = {"accuracy": accuracy, "precision": 0.91, "recall": 0.87, "f1": 0.89}
    save_model_with_metadata(model, config.model, metrics, cv_score, classification_report="Reporte", config=config)
    ```

---

## Consejos para Colaboradores

### Mejores Prácticas
1. **Modularidad**: Mantener funciones separadas para tareas específicas.
2. **Documentación**: Asegurarse de que cada función tenga una descripción clara.
3. **Evaluación**: Revisar siempre las métricas de validación cruzada antes de guardar un modelo.

### Depuración de Errores
- Si aparece el `SettingWithCopyWarning`, consulta las funciones en `preprocessing.py` y usa `.loc` para modificar datos.
- Verifica que los archivos CSV tengan el encoding correcto antes de cargarlos.

---

## Contacto
Para dudas o sugerencias, contactar a alan.dreszman@gmail.com


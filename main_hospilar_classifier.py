## main_hospital_classifier.py

'''


'''


from dataclasses import dataclass
from ydata_profiling import ProfileReport
from preprocessing import load_and_read_data
from preprocessing import crear_tipo_episodio
from preprocessing import default_clean
from preprocessing import create_final_dataframe
from preprocessing import check_final_data
from preprocessing import prepare_test_train
from modeling import evaluate_models
from modeling import train_and_evaluate_model
from evaluation import full_metrics_eval
from evaluation import save_model_with_metadata


## Definicion de variables a utilizar
@dataclass
class Config:

    archivos=['Episodios_Diagnosticos.csv',     # Nombre de los archivos que contienen la información a procesar
              'Estudios_Complementarios.csv',
              'Pacientes.csv',
              'Signos_Vitales.csv']

    path = ""    # Ruta donde se encuentran los archivos

    n_charenc=10000   # cantidad de caracteres para evaluar el encoding de los archivos

    features=['EDAD',                     # features a considerar para el modelo (chequear que los nombres correspondan a features en df_final)
              'SEXO', 
              'CANTIDAD_EPISODIOS', 
              'CANTIDAD_ESTUDIOS', 
              'CANTIDAD_SIGNOS_VITALES',
              'PRIMER_AREA_FRECUENTE',
              'ULTIMO_AREA_FRECUENTE',
              'TIPO_DIAGNOSTICO_FRECUENTE'
             ] 
    
    test_size=0.20     # proporcion del dataset de testing
    seed=42            # semilla de randomizacion

    model="Random Forest"  # modelo a utilizar en el entranmiento y evaluacion 
    cv_split=10            # segmentacion para el proceso de cross-validation


config=Config()


# Evaluacion, lectura y carga de datos

dfs = load_and_read_data(file_names=config.archivos, n_charenc=config.n_charenc)


# Division de dataframes para cada archivo 

df_episodios = dfs[0]
df_estudios = dfs[1]
df_pacientes = dfs[2]
df_signos = dfs[3]


# Agregar 'CLASE' al df de Episodios en base a 'TIPO_EPISODIO' ('CLASE'=1 si 'TIPO_EPISODIO' = 'H',  'CLASE' = 0 en otro caso)
df_episodios = crear_tipo_episodio(df_episodios)

# Limpieza inicial, especificada en la documentación: Eliminacion de registros con 'TIPO_EPISODIO'='*' en df_episodios, 
#                                                     Eliminación de registros ducplicados en df_pacientes,
#                                                     Eliminacion de columno 'ID_ITEM' en df_estudios. 
df_episodios, df_pacientes, df_estudios = default_clean(df_episodios, df_pacientes, df_estudios)

X_train, X_test, y_train, y_test = prepare_test_train(df_final,features=config.features, test_size=config.test_size, seed=config.seed)

df_evaluate_models, best_model = evaluate_models(X_train, X_test, y_train, y_test, test_size=config.test_size, random_state=config.seed)

# Agrego el mejor modelo al archivo de configuracion
config.model = best_model

model, accuracy, precision, recall, f1, auc = train_and_evaluate_model(X_train, X_test, y_train, y_test, model=config.model, seed=config.seed)

cv_score, y_pred, cm, cr=full_metrics_eval(model, X_train, y_train, X_test, y_test, cv_split=config.cv_split)


metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,     
    "f1": f1,
    "auc": auc
}


# Ajuste por hiperparametros

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

param_grid = {
    'n_estimators': [100, 200, 300],         # Número de árboles en el bosque
    'max_depth': [10, 20, 30, None],        # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],        # Mínimas muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],          # Mínimas muestras en cada hoja
    'max_features': ['sqrt', 'log2', None]  # Máximas características consideradas en cada división
}


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                   # Validación cruzada de 5 divisiones
    scoring='accuracy',     # Métrica obj
    verbose=2,              # Que detalle el progreso
    n_jobs=-1               # Usar todos los núcleos
)

grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros y el mejor score
print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor puntuación de validación cruzada:", grid_search.best_score_)


# Evaluar el modelo en los datos de prueba
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

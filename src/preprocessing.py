import pandas as pd
import chardet
import os
from ydata_profiling import ProfileReport
#import numpy as np

#######################################################################################
def load_and_read_data(file_names, path=None, default_encoding_option=1, n_charenc=10000):
    """
    Load and read multiple CSV files, attempting to detect their encodings and falling back to default encodings if necessary.

    Parameters:
    file_names (list of str): List of file names (strings) to be read.
    path (str): Path to the directory where the files are located.
    default_encoding_option (int): Numeric code representing the default encoding to try if detection fails.
    n_charenc (int): Number of characters to be red for each file in the encoding detection process (default=10000)

    Returns:
    list of pd.DataFrame: A list of DataFrames, one for each successfully read file.
    """

    if path is None:
        path = os.getcwd()
    
    # Define a mapping of encoding options
    encoding_options = {
        1: "ISO-8859-1",
        2: "ASCII",
        3: "utf-8",
        4: "utf-16",
    }

    # Validate default encoding
    if default_encoding_option not in encoding_options:
        raise ValueError(f"Invalid default encoding option. Choose from {list(encoding_options.keys())}.")

    default_encoding = encoding_options[default_encoding_option]
    
    dataframes = []  # To store the resulting DataFrames

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        #print(file_path)
        #print(path)
        try:
            # Step 1: Detect encoding
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(n_charenc))
                detected_encoding = result['encoding']

            # Step 2: Try to read the file with detected encoding
            try:
                df = pd.read_csv(file_path, encoding=detected_encoding, sep=';')
            except Exception as e:
                print(f"Warning: Failed to read {file_name} with detected encoding ({detected_encoding}). Error: {e}")

                # Step 3: Try to read with default encoding
                try:
                    df = pd.read_csv(file_path, encoding=default_encoding, sep=';')
                except Exception as e:
                    print(f"Error: Failed to read {file_name} with default encoding ({default_encoding}). Skipping file. Error: {e}")
                    continue

            # Add DataFrame to the list
            dataframes.append(df)
            print(f"Successfully read {file_name} with encoding: {default_encoding or detected_encoding}")

        except FileNotFoundError:
            print(f"Error: File {file_name} not found in path {path}.")
        except Exception as e:
            print(f"Error: Unexpected error occurred while reading {file_name}. Error: {e}")

    # Print the number of DataFrames loaded
    print(f"Total DataFrames loaded: {len(dataframes)}")

    return dataframes

# Example usage:
# file_list = ["file1.csv", "file2.csv", "file3.csv"]
# path_to_files = "./data"
# dfs = load_and_read_data(file_list, path_to_files, default_encoding_option=1)


###################################################################################################################
###################################################################################################################
def crear_tipo_episodio(df_episodios):
    
    # Crear variable objetivo (CLASE)
    df_episodios['CLASE'] = df_episodios['TIPO_EPISODIO'].apply(lambda x: 1 if x == 'H' else 0)

    return df_episodios

###################################################################################################################
###################################################################################################################
def default_clean(df_episodios, df_pacientes, df_estudios):
    """
    Realiza limpiezas básicas en los DataFrames de episodios, pacientes y estudios.

    - Elimina registros con '*' en el DataFrame de episodios.
    - Elimina duplicados en el DataFrame de pacientes.
    - Elimina la columna TIPO_EPISODIO del DataFrame de estudios.

    Args:
        df_episodios (pd.DataFrame): DataFrame con datos de episodios.
        df_pacientes (pd.DataFrame): DataFrame con datos de pacientes.
        df_estudios (pd.DataFrame): DataFrame con datos de estudios.

    Returns:
        tuple: Tres DataFrames limpios (df_episodios, df_pacientes, df_estudios).
    """
    # Limpiar episodios eliminando registros con '*'
    df_episodios = df_episodios[df_episodios['TIPO_EPISODIO'] != '*']
    
    # Eliminar duplicados en el DataFrame de pacientes
    df_pacientes = df_pacientes.drop_duplicates()
    
    # Eliminar la columna TIPO_EPISODIO en el DataFrame de estudios si existe
    if 'ID_ITEM' in df_estudios.columns:
        df_estudios = df_estudios.drop(columns=['ID_ITEM'])
    
    return df_episodios, df_pacientes, df_estudios

######################################################################################################################
######################################################################################################################

from ydata_profiling import ProfileReport

def generate_df_report(df, title="Profiling Report"):
    """
    Muestra un reporte detallado del DataFrame utilizando ydata-profiling.
    Esta función genera un informe interactivo del DataFrame, proporcionando una visión general de los datos, 
    estadísticas descriptivas, identificación de valores nulos, duplicados, distribución de variables, 
    correlaciones entre columnas y más. Este informe es útil para el análisis exploratorio de datos

    Args:
        df (pd.DataFrame): El DataFrame que se desea analizar.
        title (str, optional): El título del reporte. Por defecto, es "Profiling Report".

    Returns:
        ProfileReport: Un objeto de reporte que se puede visualizar en un Jupyter Notebook o guardar como archivo HTML.
    """

    profile = ProfileReport(df, title=title)
    
    return profile



########################################################################################################################
########################################################################################################################

def agregar_areas_frecuentes(df_episodios, df_final):
    """
    Agrega al DataFrame df_final dos columnas con el valor más frecuente
    de 'primer_area' y 'ultima_area' para cada paciente en df_episodios.

    Args:
        df_episodios (pd.DataFrame): DataFrame con información de episodios (incluye 'Paciente', 'primer_area', 'ultima_area').
        df_final (pd.DataFrame): DataFrame con información agregada por paciente.

    Returns:
        pd.DataFrame: DataFrame df_final con las columnas 'primer_area_frecuente' y 'ultima_area_frecuente' agregadas.
    """
    # Calcular la "primer_area" más frecuente para cada paciente
    primer_area_frecuente = (
        df_episodios.groupby('PACIENTE')['PRIMER_AREA']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={'PRIMER_AREA': 'PRIMER_AREA_FRECUENTE'})
    )
    
    # Calcular la "ultima_area" más frecuente para cada paciente
    ultima_area_frecuente = (
        df_episodios.groupby('PACIENTE')['ULTIMO_AREA']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={'ULTIMO_AREA': 'ULTIMO_AREA_FRECUENTE'})
    )
    
    # Unir la información al df_final
    df_final = df_final.merge(primer_area_frecuente, on='PACIENTE', how='left')
    df_final = df_final.merge(ultima_area_frecuente, on='PACIENTE', how='left')
    
    return df_final

########################################################################################################################
########################################################################################################################
def agregar_tipo_diagnostico_frecuente(df_episodios, df_final):
    """
    Agrega al DataFrame df_final una columna con el valor más frecuente
    de 'tipo_diagnostico' para cada paciente en df_episodios.

    Args:
        df_episodios (pd.DataFrame): DataFrame con información de episodios (incluye 'Paciente', 'tipo_diagnostico').
        df_final (pd.DataFrame): DataFrame con información agregada por paciente.

    Returns:
        pd.DataFrame: DataFrame df_final con la columna 'tipo_diagnostico' agregada.
    """
    # Calcular la "tipo_diagnostico" más frecuente para cada paciente
    tipo_diagnostico_frecuente = (
        df_episodios.groupby('PACIENTE')['TIPO_DIAGNOSTICO']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={'TIPO_DIAGNOSTICO': 'TIPO_DIAGNOSTICO_FRECUENTE'})
    )
    
    # Calcular la "ultima_area" más frecuente para cada paciente
    #ultima_area_frecuente = (
    #    df_episodios.groupby('PACIENTE')['ULTIMO_AREA']
    #    .agg(lambda x: x.value_counts().idxmax())
    #    .reset_index()
    #    .rename(columns={'ULTIMO_AREA': 'ULTIMO_AREA_FRECUENTE'})
    #)
    
    # Unir la información al df_final
    df_final = df_final.merge(tipo_diagnostico_frecuente, on='PACIENTE', how='left')
    #df_final = df_final.merge(tipo_diagnostico_frecuente, on='PACIENTE', how='left')
    
    return df_final

########################################################################################################################
########################################################################################################################





def create_final_dataframe(df_pacientes, df_episodios, df_signos, df_estudios):
    '''
    Crea un dataframe que concatena los features relevados previamente
    como utiles para realizar posteriormente el entrenamiento

    Args:
        df_episodios (pd.DataFrame): DataFrame con datos de episodios.
        df_pacientes (pd.DataFrame): DataFrame con datos de pacientes.
        df_estudios (pd.DataFrame): DataFrame con datos de estudios.
        df_signos (pd.DataFrame): DataFrame con datos de signos vitales.

    Returns:
        df_final (pd.DataFrame): DataFrame con información agregada por paciente.

    '''
    
    # Crear el dataframe final con EDAD y SEXO
    df_final = df_pacientes[['PACIENTE', 'EDAD', 'SEXO']]
    
    # Codificar SEXO (F -> 0, M -> 1)
    #df_final['SEXO'] = df_final['SEXO'].map({'F': 0, 'M': 1})
    df_final.loc[:, 'SEXO'] = df_final['SEXO'].map({'F': 0, 'M': 1})
    
    # Agregar cantidad de internaciones por paciente
    internaciones_agg = df_episodios[df_episodios['TIPO_EPISODIO'] == 'H'].groupby('PACIENTE').size()
    df_final = pd.merge(df_final, internaciones_agg.rename('CANTIDAD_INTERNACIONES'), on='PACIENTE', how='left')
    
    # Agregar cantidad de episodios por paciente
    episodios_agg = df_episodios.groupby('PACIENTE').size()
    df_final = pd.merge(df_final, episodios_agg.rename('CANTIDAD_EPISODIOS'), on='PACIENTE', how='left')
    
    # Agregar cantidad de estudios por paciente
    estudios_agg = df_estudios.groupby('PACIENTE').size()
    df_final = pd.merge(df_final, estudios_agg.rename('CANTIDAD_ESTUDIOS'), on='PACIENTE', how='left')
    
    # Agregar cantidad de signos vitales por paciente
    signos_agg = df_signos.groupby('PACIENTE').size()
    df_final = pd.merge(df_final, signos_agg.rename('CANTIDAD_SIGNOS_VITALES'), on='PACIENTE', how='left')
    
    # Rellenar valores faltantes con 0 (pacientes que no tienen internaciones, estudios, etc.)
    df_final.fillna(0, inplace=True)
    
    #from preprocessing import agregar_areas_frecuentes
    
    df_final = agregar_areas_frecuentes(df_episodios, df_final)

    
    df_final = agregar_tipo_diagnostico_frecuente(df_episodios, df_final)

    
    df_final['TIPO_EPISODIO'] = (df_final['CANTIDAD_INTERNACIONES'] > 0).astype(int)
    
    df_final = df_final.drop(columns=['CANTIDAD_INTERNACIONES'])

    return df_final

########################################################################################################################
########################################################################################################################
def check_final_data(df_final,df_pacientes):
    '''
    Asegura que la cantidad de registros en el dataframe final coincide con la longitud
    del dataframe que contiene los datos sobre los que se agrego informacion relativa a
    features para realizar el entrenamiento. Asegura que los agrupamientos realizados 
    hayan sido los correctos.

    Args:
        df_pacientes (pd.DataFrame): DataFrame con datos de pacientes.
        df_final (pd.DataFrame): DataFrame con información agregada por paciente.
    
    '''
   
    #assert(len(df_final)==len(df_pacientes), "Hay una discrepancia en la cantidad de registros del dataframe final, no es la cantidad correcta")
    if len(df_final) != len(df_pacientes):
        raise ValueError(f"Hay una discrepancia en la cantidad de registros del dataframe final ({len(df_final)}), no coincide con el total de pacientes ({len(df_pacientes)}).")
    
    #print(f"La cantidad de registros es la correcta: total de {len(df_final)}")

    print("La cantidad de registros es la correcta: total de ", len(df_final))

    

########################################################################################################################
########################################################################################################################
def prepare_test_train(df_final,features=['EDAD', 'SEXO', 'CANTIDAD_EPISODIOS', 'CANTIDAD_ESTUDIOS', 'CANTIDAD_SIGNOS_VITALES','PRIMER_AREA_FRECUENTE','ULTIMO_AREA_FRECUENTE'],test_size=0.20,seed=42):
    '''
    Devuelve los conjuntos de datos para train y para test.

    Args:
        - df_final (pd.DataFrame): DataFrame con información por paciente con todos 
          los features agregados/generados en create_final_dataframe()

        - features [string](opcional): lista de strings. Todos los strings deben coincidir con los nombres de los features en df_final
          si no se especifica se utilizan todos los features presentes.
          
        - test_size (float): valor entre 0-1. Proporción del dataset que se utilizará para la etapa de test.
        - seed (int): valor semilla para aleatorizar el proceso de división de datos en train/test.

    Return:
        - X_train (pd.DataFrame): Conjunto de datos utilizado para la etapa de de entrenamiento  
        - X_test  (pd.DataFrame): Valores target para la etapa de entrenamiento
        - y_train (pd.DataFrame): Conjunto de datos utilizado para la etapa de de test
        - y_test  (pd.DataFrame): Valores target para la etapa de test
        
    '''
    #X = df_final[['EDAD', 'SEXO', 'CANTIDAD_EPISODIOS', 'CANTIDAD_ESTUDIOS', 'CANTIDAD_SIGNOS_VITALES','PRIMER_AREA_FRECUENTE','ULTIMO_AREA_FRECUENTE']]
    X = df_final[features]
    y = df_final['TIPO_EPISODIO']
    
    X = pd.get_dummies(X, drop_first=True)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

########################################################################################################################
########################################################################################################################




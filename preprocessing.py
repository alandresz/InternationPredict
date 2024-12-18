import pandas as pd
import chardet
import os

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

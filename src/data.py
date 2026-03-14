import os
import pandas as pd
import kagglehub

def first_time_setup():

    # Download latest version
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

    print("Path to dataset files:", path)

    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    print(files)

    data = {}

    for file in files:
        name = file.replace(".csv", "")  # remove .csv for dict key
        data[name] = pd.read_csv(os.path.join(path, file))

    # Check what’s loaded
    print(data.keys())

    # Return the path and the data dictionary
    return path, data

import pandas as pd
import os

def load_data(path):
    """
    Load all 9 Olist CSV files from a given folder into a dictionary.

    Args:
        path (str): Local folder containing the CSV files.

    Returns:
        dict: Dictionary where keys are filenames (without .csv) and values are pandas DataFrames.
    """
    # List all CSV files
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    
    # Dictionary to store dataframes
    data = {}
    
    for file in files:
        name = file.replace(".csv", "")  # use filename as key
        data[name] = pd.read_csv(os.path.join(path, file))
    
    # Optional: print summary
    print("Datasets loaded:", list(data.keys()))
    
    return data
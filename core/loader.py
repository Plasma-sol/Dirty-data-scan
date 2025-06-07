import pandas as pd
import pathlib
import numpy as np

def load_data(file_path):
    """
    Load data from a file path. Supports CSV, Excel, and JSON formats.
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    file_path = pathlib.Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_path.suffix == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
    

def get_column_names(df: pd.DataFrame) -> list[str]:

    """
    Get the names of the columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        list[str]: List of column names.
    """
    return df.columns.tolist() if df is not None else []



def filter_data(df: pd.DataFrame, remove_column: list[str] = []) -> pd.DataFrame:
    """
    Filter out a specific column from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        remove_column (str): Column name to be removed.
        
    Returns:
        pd.DataFrame: DataFrame with the specified column removed.
    """
    if remove_column is None or len(remove_column) == 0:
        return df
    
    for column in remove_column:
        if column in df.columns:
            df = df.drop(columns=column)
 
    return df.reset_index(drop=True)


def standardise_nans(df, nan_type=pd.NA):

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # List of values to be considered as NA
    na_values = [
        None,           # Python None
        np.nan,         # NumPy NaN
        pd.NA,
        '',            # Empty string
        'nan',         # String 'nan'
        'NaN',         # String 'NaN'
        'NA',          # String 'NA'
        'null',        # String 'null'
        'NULL',        # String 'NULL'
        'None'         # String 'None'
    ]
    
    # Replace all NA-like values with pd.NA
    for column in df.columns:
        # For numeric columns, first convert np.nan to pd.NA
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].replace({np.nan: nan_type})
        
        # For all columns, replace various NA values
        df[column] = df[column].replace(na_values, nan_type)
        
        # Handle empty strings in object/string columns
        if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].replace(r'^\s*$', nan_type, regex=True)
    
    return df

# def nan_standardisation(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Standardize NaN values in the DataFrame.
    
#     Args:
#         df (pd.DataFrame): Input DataFrame.
        
#     Returns:
#         pd.DataFrame: DataFrame with NaN values standardized.
#     """
#     if df is None or df.empty:
#         return df
    
#     # Replace NaN with a standard value, e.g., 'Unknown' for object types
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = df[column].fillna('Unknown')
    
#     # Replace NaN with 0 for numeric types
#     for column in df.select_dtypes(include=['number']).columns:
#         df[column] = df[column].fillna(0)
    
#     return df.reset_index(drop=True)
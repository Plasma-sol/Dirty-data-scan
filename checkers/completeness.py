import pandas as pd
def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the completeness of a DataFrame by calculating the percentage 
    of non-empty cells for entire dataset. 
    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        pd.DataFrame: A DataFrame containing the percentage of missing values for each column.
    """
    column_empty_cells = df.isna().sum() + (df == '').sum()
    return column_empty_cells

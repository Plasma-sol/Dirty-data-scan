import pandas as pd
import numpy as np



# def checker(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Perform a series of checks on the DataFrame and return a summary DataFrame.
    
#     Args:
#         df (pd.DataFrame): The DataFrame to check.

#     Returns:
#         pd.DataFrame: A DataFrame containing the results of the checks.
#     """
#     completeness_result = completeness(df)
#     duplicate_result = dupelicate(df)
    
#     result = pd.concat([completeness_result, duplicate_result], axis=1)
#     result.columns = ['missing_values', 'duplicate_count']
    
#     return result

class Checker():
    def __init__(self):
        pass
    @staticmethod
    def completeness(df: pd.DataFrame) -> pd.DataFrame:
        """
        Check the completeness of a DataFrame by calculating the percentage 
        of non-empty cells for entire dataset. 
        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            pd.DataFrame: A DataFrame containing the percentage of missing values for each column.
        """
        column_full_percent = 1 - ((df.isna().sum() + (df == '').sum())/ len(df))
        return column_full_percent
    
    @staticmethod
    def dupelicate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for duplicate rows in a DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to check for duplicates.

        Returns:
            pd.DataFrame: A DataFrame containing the count of duplicate rows.
        """
        duplicate_count = df.duplicated().sum()
        return pd.DataFrame({'duplicate_count': [duplicate_count]})
    
    def run(self, df) -> pd.DataFrame:
        """
        Run the checker on the DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame containing the results of the checks.
        """
        completeness = self.completeness(df)
        duplicate = self.dupelicate(df)
        result = pd.concat([completeness, duplicate], axis=1)
        result.columns = ['missing_values', 'duplicate_count']
        return result

if __name__ == "__main__":
    df = pd.read_csv('../examples/test1.csv')
    checker = Checker()
    result = checker.run(df)
    print(result)


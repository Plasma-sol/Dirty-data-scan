import pandas as pd
import numpy as np
from scipy.stats import chi2


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
    
    @staticmethod
    def mcar_test(df: pd.DataFrame, alpha: float = 0.05) -> bool:
        """
        Perform the MCAR test on a DataFrame to check if the missing data is 
        missing completely at random.
        
        Args:
            df (pd.DataFrame): The DataFrame to test.
            alpha (float): Significance level for the test.

        Returns:
            bool: True if the data is MCAR, False otherwise.
        """
        p_m = df.isnull().mean()
        # Calculate the proportion of complete cases for each variable
        p_c = df.dropna().shape[0] / df.shape[0]
        # Calculate the correlation matrix for all pairs of variables that have complete cases
        R_c = df.dropna().corr()
        # Calculate the correlation matrix for all pairs of variables using all observations
        R_all = df.corr()
        # Calculate the difference between the two correlation matrices
        R_diff = R_all - R_c
        # Calculate the variance of the R_diff matrix
        V_Rdiff = np.var(R_diff, ddof=1)
        # Calculate the expected value of V_Rdiff under the null hypothesis that the missing data is MCAR
        E_Rdiff = (1 - p_c) / (1 - p_m).sum()
        # Calculate the test statistic
        T = np.trace(R_diff) / np.sqrt(V_Rdiff * E_Rdiff)
        # Calculate the degrees of freedom
        df = df.shape[1] * (df.shape[1] - 1) / 2
        # Calculate the p-value using a chi-squared distribution with df degrees of freedom and the test statistic T
        p_value = 1 - chi2.cdf(T ** 2, df)
        # Create a matrix of missing values that represents the pattern of missingness in the dataset
        missingness_matrix = df.isnull().astype(int)
        # Return the missingness matrix and the p-value
        return missingness_matrix, p_value


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
    # print(Checker.mcar_test(df))
    # result.to_csv('../examples/checker_result.csv', index=False)

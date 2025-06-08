import pandas as pd
import numpy as np
import pickle
from scipy.stats import chi2
from scipy import stats


# from checkers.correlation import correlation_matrix

class Checker():
    def __init__(self):
        pass

    def analyze_outliers(self, df, cols=None, threshold=3):
        """
        Analyze outliers in the dataset and return percentage of outliers per row and column.
    
        Parameters:
        df (pd.DataFrame): Input dataframe
        cols (list): List of column names to check. If None, uses all numeric columns.
        threshold (float): Z-score threshold for outlier detection (default: 3)
    
        Returns:
        dict: Contains outlier analysis results
        """
        df_copy = df.copy()
    
        # If no columns specified, use all numeric columns
        if cols is None:
            cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
        print(f"Analyzing outliers in columns: {cols}")
        print(f"Using threshold: {threshold} standard deviations\n")
    
        # Calculate z-scores for specified columns
        z_scores = np.abs(stats.zscore(df_copy[cols], nan_policy='omit'))
        z_scores_df = pd.DataFrame(z_scores, columns=cols, index=df_copy.index)
    
        # Create outlier mask (True = outlier)
        outlier_mask = z_scores_df > threshold
    
        # Calculate percentage of outliers per column
        outliers_per_column = (outlier_mask.sum() / len(df_copy)) * 100
    
        # Calculate percentage of outliers per row (across specified columns)
        outliers_per_row = (outlier_mask.sum(axis=1) / len(cols)) * 100
    
        # Count rows with at least one outlier
        rows_with_outliers = (outlier_mask.any(axis=1)).sum()
        rows_with_outliers_pct = (rows_with_outliers / len(df_copy)) * 100
    
        # Summary statistics
        total_values = len(df_copy) * len(cols)
        total_outliers = outlier_mask.sum().sum()
        overall_outlier_pct = (total_outliers / total_values) * 100
        
        # Calculate average z-scores of outliers
        # Get z-scores only for outlier points
        outlier_z_scores = z_scores_df[outlier_mask]
        
        # Calculate overall average z-score of all outliers
        all_outlier_z_scores = []
        avg_z_scores_per_column = {}
        
        for col in cols:
            col_outlier_z_scores = z_scores_df.loc[outlier_mask[col], col]
            if len(col_outlier_z_scores) > 0:
                avg_z_scores_per_column[col] = col_outlier_z_scores.mean()
                all_outlier_z_scores.extend(col_outlier_z_scores.tolist())
            else:
                avg_z_scores_per_column[col] = 0.0
        
        overall_avg_outlier_z_score = np.mean(all_outlier_z_scores) if all_outlier_z_scores else 0.0
        max_outlier_z_score = np.max(all_outlier_z_scores) if all_outlier_z_scores else 0.0
        min_outlier_z_score = np.min(all_outlier_z_scores) if all_outlier_z_scores else 0.0
    
        # Print analysis
        print("=== OUTLIER ANALYSIS RESULTS ===")
        print(f"Total observations: {len(df_copy)}")
        print(f"Total columns analyzed: {len(cols)}")
        print(f"Overall outlier percentage: {overall_outlier_pct:.2f}%")
        print(f"Rows with at least one outlier: {rows_with_outliers} ({rows_with_outliers_pct:.2f}%)")
        
        # Calculate average of the per-column averages
        column_averages = [avg for avg in avg_z_scores_per_column.values() if avg > 0]
        avg_of_column_averages = np.mean(column_averages) if column_averages else 0.0
        
        # Print outlier z-score statistics
        print(f"\n--- Outlier Z-Score Statistics ---")
        print(f"Total outlier datapoints: {len(all_outlier_z_scores)}")
        print(f"Overall average z-score of outliers: {overall_avg_outlier_z_score:.2f}")
        print(f"Average of per-column averages: {avg_of_column_averages:.2f}")
        print(f"Range of outlier z-scores: {min_outlier_z_score:.2f} - {max_outlier_z_score:.2f}")
    
        print("\n--- Outliers per Column ---")
        for col in cols:
            count = outlier_mask[col].sum()
            pct = outliers_per_column[col]
            avg_z = avg_z_scores_per_column[col]
            if count > 0:
                print(f"{col}: {count} outliers ({pct:.2f}%), avg z-score: {avg_z:.2f}")
            else:
                print(f"{col}: {count} outliers ({pct:.2f}%)")
    
        print(f"\n--- Outliers per Row (showing top 10) ---")
        top_outlier_rows = outliers_per_row[outliers_per_row > 0].sort_values(ascending=False).head(10)
        if len(top_outlier_rows) > 0:
            for idx, pct in top_outlier_rows.items():
                count = outlier_mask.loc[idx].sum()
                # Show the actual z-scores for this row's outliers
                row_outlier_z_scores = z_scores_df.loc[idx][outlier_mask.loc[idx]]
                avg_row_z = row_outlier_z_scores.mean() if len(row_outlier_z_scores) > 0 else 0
                print(f"Row {idx}: {count}/{len(cols)} columns are outliers ({pct:.2f}%), avg z-score: {avg_row_z:.2f}")
        else:
            print("No rows with outliers found.")
    
        # Return detailed results
        results = {
            'outlier_mask': outlier_mask,
            'z_scores': z_scores_df,
            'outliers_per_column': outliers_per_column,
            'outliers_per_row': outliers_per_row,
            'rows_with_outliers': rows_with_outliers,
            'rows_with_outliers_pct': rows_with_outliers_pct,
            'overall_outlier_pct': overall_outlier_pct,
            'total_outliers': total_outliers,
            'columns_analyzed': cols,
            'overall_avg_outlier_z_score': overall_avg_outlier_z_score,
            'avg_z_scores_per_column': avg_z_scores_per_column,
            'min_outlier_z_score': min_outlier_z_score,
            'max_outlier_z_score': max_outlier_z_score,
            'avg_of_column_averages': avg_of_column_averages,
            'total_outlier_datapoints': len(all_outlier_z_scores)
        }
    
        return results 
    
    def get_all_correlation_pairs(self, data, method='pearson', round_digits=3, min_periods=1):
        """
        Generate a list of all variable pairs with their correlation coefficients and p-values.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data to compute correlations
        method : str, default 'pearson'
            Correlation method: 'pearson' or 'spearman'
        round_digits : int, default 3
            Number of decimal places to round to
        min_periods : int, default 1
            Minimum number of observations required per pair
            
        Returns:
        --------
        list: List of dictionaries containing variable pairs, correlations, and p-values
        """
        try:
            from scipy.stats import pearsonr, spearmanr
        except ImportError:
            raise ImportError("scipy is required for p-values calculation.")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['float64', 'float32', 'float16'])
        
        # If no float columns, try broader numeric selection
        if numeric_data.empty:
            numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found in DataFrame")
        
        columns = numeric_data.columns
        n = len(columns)
        
        # List to store all pairs
        correlation_pairs = []
        
        # Compute correlations and p-values for all pairs
        for i in range(n):
            for j in range(i+1, n):  # Only upper triangle to avoid duplicates
                col1_name = columns[i]
                col2_name = columns[j]
                
                # Get valid (non-NaN) pairs
                col1 = numeric_data.iloc[:, i]
                col2 = numeric_data.iloc[:, j]
                
                # Remove NaN pairs
                valid_mask = ~(pd.isna(col1) | pd.isna(col2))
                valid_col1 = col1[valid_mask]
                valid_col2 = col2[valid_mask]
                
                # Check if enough valid observations
                if len(valid_col1) < min_periods:
                    corr_val = np.nan
                    p_val = np.nan
                    n_obs = len(valid_col1)
                else:
                    if method == 'pearson':
                        corr_val, p_val = pearsonr(valid_col1, valid_col2)
                    elif method == 'spearman':
                        corr_val, p_val = spearmanr(valid_col1, valid_col2)
                    else:
                        raise ValueError("Method must be 'pearson' or 'spearman'")
                    
                    n_obs = len(valid_col1)
                
                # Round values if specified
                if round_digits is not None and not np.isnan(corr_val):
                    corr_val = round(corr_val, round_digits)
                    p_val = round(p_val, round_digits)
                
                # Add to results
                correlation_pairs.append({
                    'variable_1': col1_name,
                    'variable_2': col2_name,
                    'correlation': corr_val,
                    'p_value': p_val,
                    'n_observations': n_obs,
                    'method': method
                })
        
        return correlation_pairs

    def print_correlation_pairs(self, correlation_pairs, show_details=True):
        """
        Print correlation pairs in a clean format.
        
        Parameters:
        -----------
        correlation_pairs : list
            List of correlation pair dictionaries
        show_details : bool, default True
            Whether to show additional details like n_observations and method
        """
        print(f"Found {len(correlation_pairs)} variable pairs:")
        print("-" * 80)
        
        for i, pair in enumerate(correlation_pairs, 1):
            var1 = pair['variable_1']
            var2 = pair['variable_2']
            corr = pair['correlation']
            p_val = pair['p_value']
            
            if np.isnan(corr):
                corr_str = "NaN"
                p_str = "NaN"
            else:
                corr_str = f"{corr:.3f}"
                p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "NaN"
            
            print(f"{i:2d}. {var1} - {var2}")
            print(f"    Correlation: {corr_str}")
            print(f"    P-value:     {p_str}")
            
            if show_details:
                print(f"    N obs:       {pair['n_observations']}")
                print(f"    Method:      {pair['method']}")
            
            print()

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
    def duplicate(df: pd.DataFrame) -> pd.DataFrame:
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
        dimentions = df.shape
        completeness = self.completeness(df)
        duplicate = self.duplicate(df)
        outliers = self.analyze_outliers(df)
        corr = self.get_all_correlation_pairs(df)
        return dimentions, completeness, duplicate, outliers, corr

if __name__ == "__main__":
    df = pd.read_csv('../../examples/test1.csv')
    checker = Checker()
    data = checker.run(df)
    with open('../../examples/dimensions.txt', 'w') as f:
        f.write(f"Dimensions of the DataFrame: {data[0]}\n")
    data[1].to_csv('../../examples/completeness.csv', index=False)
    data[2].to_csv('../../examples/duplicates.csv', index=False)
    with open('../../examples/outliers.pkl', 'wb') as f:
        pickle.dump(data[3], f)
    with open('../../examples/correlation.pkl', 'wb') as f:
        pickle.dump(data[4], f)
    # complete_and_duplicates = checker.run(df)
    # outliers = checker.analyze_outliers(df)
    # corr = checker.get_all_correlation_pairs(df)
    # checker.print_correlation_pairs(corr)
    # print(corr)
    # print(complete_and_duplicates)
    # print(Checker.mcar_test(df))
    # result.to_csv('../examples/checker_result.csv', index=False)

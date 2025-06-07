import pandas as pd
import numpy as np
from scipy import stats

def analyze_outliers(df, cols=None, threshold=3):
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
    
    # Print analysis
    print("=== OUTLIER ANALYSIS RESULTS ===")
    print(f"Total observations: {len(df_copy)}")
    print(f"Total columns analyzed: {len(cols)}")
    print(f"Overall outlier percentage: {overall_outlier_pct:.2f}%")
    print(f"Rows with at least one outlier: {rows_with_outliers} ({rows_with_outliers_pct:.2f}%)\n")
    
    print("--- Outliers per Column ---")
    for col in cols:
        count = outlier_mask[col].sum()
        pct = outliers_per_column[col]
        print(f"{col}: {count} outliers ({pct:.2f}%)")
    
    print(f"\n--- Outliers per Row (showing top 10) ---")
    top_outlier_rows = outliers_per_row[outliers_per_row > 0].sort_values(ascending=False).head(10)
    if len(top_outlier_rows) > 0:
        for idx, pct in top_outlier_rows.items():
            count = outlier_mask.loc[idx].sum()
            print(f"Row {idx}: {count}/{len(cols)} columns are outliers ({pct:.2f}%)")
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
        'columns_analyzed': cols
    }
    
    return results

def remove_outliers(df, cols=None, threshold=3, method='any_column'):
    """
    Remove outliers from the dataset based on specified method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    cols (list): List of column names to check. If None, uses all numeric columns.
    threshold (float): Z-score threshold for outlier detection (default: 3)
    method (str): 'any_column' or 'by_column' or 'iqr'
    
    Returns:
    pd.DataFrame: Cleaned dataframe with outliers removed
    """
    df_copy = df.copy()
    
    # If no columns specified, use all numeric columns
    if cols is None:
        cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    original_shape = df_copy.shape
    
    if method == 'any_column':
        # Remove rows where ANY specified column has outliers
        z_scores = np.abs(stats.zscore(df_copy[cols], nan_policy='omit'))
        keep_rows = (z_scores <= threshold).all(axis=1)
        df_clean = df_copy[keep_rows]
        
    elif method == 'by_column':
        # Remove outliers column by column
        df_clean = df_copy.copy()
        for col in cols:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                mask = np.abs(df_clean[col] - mean_val) <= threshold * std_val
                df_clean = df_clean[mask]
                
    elif method == 'iqr':
        # Remove outliers using IQR method
        df_clean = df_copy.copy()
        factor = 1.5 if threshold == 3 else threshold/2  # Adjust IQR factor based on threshold
        
        for col in cols:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
    else:
        raise ValueError("Method must be 'any_column', 'by_column', or 'iqr'")
    
    # Print removal summary
    removed_rows = original_shape[0] - df_clean.shape[0]
    removal_pct = (removed_rows / original_shape[0]) * 100
    
    print(f"\n=== OUTLIER REMOVAL RESULTS ===")
    print(f"Method used: {method}")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df_clean.shape}")
    print(f"Rows removed: {removed_rows} ({removal_pct:.2f}%)")
    
    if method == 'any_column':
        removed_indices = df_copy.index.difference(df_clean.index).tolist()
        if len(removed_indices) > 0 and len(removed_indices) <= 10:
            print(f"Removed row indices: {removed_indices}")
        elif len(removed_indices) > 10:
            print(f"Removed row indices (first 10): {removed_indices[:10]}...")
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Create sample data with known outliers
    np.random.seed(42)
    df = pd.DataFrame({
        'col1': np.random.normal(50, 10, 100),
        'col2': np.random.normal(100, 20, 100),
        'col3': np.random.normal(25, 5, 100),
        'col4': np.random.normal(75, 15, 100)
    })
    
    # Add some artificial outliers
    df.loc[0, 'col1'] = 150    # Strong outlier
    df.loc[1, 'col2'] = 300    # Strong outlier  
    df.loc[2, 'col3'] = 5      # Moderate outlier
    df.loc[5, 'col1'] = 20     # Moderate outlier
    df.loc[5, 'col4'] = 150    # Same row, different column outlier
    
    print("Sample dataset created with artificial outliers.\n")

    # Step 1: Analyze outliers
    analysis = analyze_outliers(df)
    
    # Step 2: Remove outliers using different methods
    print("\n" + "="*50)
    df_clean1 = remove_outliers(df, method='any_column')
    
    print("\n" + "="*50)
    df_clean2 = remove_outliers(df, method='by_column')
    

test = pd.read_csv('https://raw.githubusercontent.com/Plasma-sol/Dirty-data-scan/refs/heads/main/examples/test1.csv?token=GHSAT0AAAAAADFIIAZZ4UTGLGPYPRIADAB22CELFCA')

analyze_outliers(test, threshold=3)
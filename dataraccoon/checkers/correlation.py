import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def get_all_correlation_pairs(data, method='pearson', round_digits=3, min_periods=1):
    """
    Generate a DataFrame of all variable pairs with their correlation coefficients and p-values.
    
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
    pandas.DataFrame: DataFrame with columns ['variable_1', 'variable_2', 'correlation', 'p_value']
    """
    # Select only numeric columns
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
            else:
                if method == 'pearson':
                    corr_val, p_val = pearsonr(valid_col1, valid_col2)
                elif method == 'spearman':
                    corr_val, p_val = spearmanr(valid_col1, valid_col2)
                else:
                    raise ValueError("Method must be 'pearson' or 'spearman'")
            
            # Round values if specified
            if round_digits is not None and not np.isnan(corr_val):
                corr_val = round(corr_val, round_digits)
                p_val = round(p_val, round_digits)
            
            # Add to results
            correlation_pairs.append({
                'variable_1': col1_name,
                'variable_2': col2_name,
                'correlation': corr_val,
                'p_value': p_val
            })
    
    return pd.DataFrame(correlation_pairs)

#test = pd.read_csv('https://raw.githubusercontent.com/Plasma-sol/Dirty-data-scan/refs/heads/main/examples/test1.csv?token=GHSAT0AAAAAADFIIAZYLEYNEQ2PXMIDODLY2CFKHBQ')
#corr = get_all_correlation_pairs(test, method='pearson', round_digits=3)

#print(corr)

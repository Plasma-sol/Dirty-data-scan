import numpy as np
import pandas as pd

def get_all_correlation_pairs(data, method='pearson', round_digits=3, min_periods=1):
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

def print_correlation_pairs(correlation_pairs, show_details=True):
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

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'Height': np.random.normal(170, 10, 100),
        'Weight': np.random.normal(70, 15, 100),
        'Age': np.random.randint(18, 80, 100),
        'Income': np.random.exponential(50000, 100),
        'Score': np.random.normal(75, 12, 100)
    })
    
    # Add some correlations to make it interesting
    data['Weight'] = data['Height'] * 0.8 + np.random.normal(0, 5, 100)
    data['Income'] = data['Age'] * 1000 + np.random.exponential(30000, 100)
    
    # Add some missing values
    data.loc[5:10, 'Age'] = np.nan
    data.loc[15:20, 'Income'] = np.nan
    
    print("Sample Data Info:")
    print(f"Shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print()
    
    # Get all correlation pairs
    pairs = get_all_correlation_pairs(data, method='pearson')
    
    # Print results
    print_correlation_pairs(pairs)
    
    # Alternative: Create a simple DataFrame for easy viewing
    pairs_df = pd.DataFrame(pairs)
    print("As DataFrame:")
    print(pairs_df[['variable_1', 'variable_2', 'correlation', 'p_value']].to_string(index=False))

test = pd.read_csv('https://raw.githubusercontent.com/Plasma-sol/Dirty-data-scan/refs/heads/main/examples/test1.csv?token=GHSAT0AAAAAADFIIAZZITZEW44V7Z2DFWEM2CENWQA')
corr = correlation_matrix(test, method='pearson', round_digits=3)

p_values = corr_with_pvalues(test, round_digits=3)
print("Correlation Matrix:")        
print(p_values[0])  # Correlation matrix
print(p_values[1])

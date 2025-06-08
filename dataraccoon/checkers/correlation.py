import numpy as np
import pandas as pd

def correlation_matrix(data, method='pearson', round_digits=3, min_periods=1):
    """
    Compute correlation matrix similar to R's cor() function with NA handling.
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.array
        Input data to compute correlations
    method : str, default 'pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'
    round_digits : int, default 3
        Number of decimal places to round to (None for no rounding)
    min_periods : int, default 1
        Minimum number of observations required per pair of columns
        to have a valid result (pandas only)
    
    Returns:
    --------
    pandas.DataFrame or numpy.array
        Correlation matrix with NaN values ignored
    """
    
    if isinstance(data, pd.DataFrame):
        # Select only float columns (most restrictive numeric type)
        float_data = data.select_dtypes(include=['float64', 'float32', 'float16'])
        
        # If no float columns, try broader numeric selection
        if float_data.empty:
            float_data = data.select_dtypes(include=[np.number])
        
        if float_data.empty:
            raise ValueError("No numeric columns found in DataFrame")
        
        # Use pandas corr method which handles NaN automatically
        corr = float_data.corr(method=method, min_periods=min_periods)
        if round_digits is not None:
            corr = corr.round(round_digits)
        return corr
    
    elif isinstance(data, np.ndarray):
        # For numpy arrays, check data types and handle NaN values
        if method != 'pearson':
            print("Warning: Only Pearson correlation available for numpy arrays")
        
        # Try to convert to float, which will handle strings that represent numbers
        try:
            numeric_data = data.astype(float)
        except (ValueError, TypeError):
            raise ValueError("Numpy array contains non-numeric data that cannot be converted to float")
        
        # Convert to DataFrame temporarily for proper NaN handling
        temp_df = pd.DataFrame(numeric_data)
        corr = temp_df.corr(min_periods=min_periods).values
        
        if round_digits is not None:
            corr = np.round(corr, round_digits)
        return corr
    
    else:
        raise TypeError("Input must be pandas DataFrame or numpy array")

def corr_with_pvalues(data, method='pearson', round_digits=3, min_periods=1, 
                     high_corr_threshold=0.95, p_threshold=0.05):
    """
    Compute correlation matrix with p-values (requires scipy), handling NaN values,
    and identify highly correlated pairs.
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.array
        Input data
    method : str, default 'pearson'
        Correlation method: 'pearson', 'spearman'
    round_digits : int, default 3
        Decimal places to round
    min_periods : int, default 1
        Minimum number of valid observations required
    high_corr_threshold : float, default 0.95
        Absolute correlation threshold for identifying high correlations
    p_threshold : float, default 0.05
        P-value threshold for significance
        
    Returns:
    --------
    dict: {
        'correlation_matrix': correlation matrix,
        'p_value_matrix': p-value matrix,
        'high_corr_summary': summary of highly correlated pairs
    }
    """
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        raise ImportError("scipy is required for p-values. Use correlation_matrix() for correlations only.")
    
    if isinstance(data, pd.DataFrame):
        # Select only float columns (most restrictive numeric type)
        float_data = data.select_dtypes(include=['float64', 'float32', 'float16'])
        
        # If no float columns, try broader numeric selection
        if float_data.empty:
            float_data = data.select_dtypes(include=[np.number])
        
        if float_data.empty:
            raise ValueError("No numeric columns found in DataFrame")
        
        columns = float_data.columns
        n = len(columns)
        
        # Initialize matrices
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))
        
        # Compute correlations and p-values
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    # Get valid (non-NaN) pairs from float data
                    col1 = float_data.iloc[:, i]
                    col2 = float_data.iloc[:, j]
                    
                    # Remove NaN pairs
                    valid_mask = ~(pd.isna(col1) | pd.isna(col2))
                    valid_col1 = col1[valid_mask]
                    valid_col2 = col2[valid_mask]
                    
                    # Check if enough valid observations
                    if len(valid_col1) < min_periods:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan
                    else:
                        if method == 'pearson':
                            corr, p_val = pearsonr(valid_col1, valid_col2)
                        elif method == 'spearman':
                            corr, p_val = spearmanr(valid_col1, valid_col2)
                        else:
                            raise ValueError("Method must be 'pearson' or 'spearman'")
                        
                        corr_matrix[i, j] = corr
                        p_matrix[i, j] = p_val
        
        # Convert to DataFrames
        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)
        
        if round_digits is not None:
            corr_df = corr_df.round(round_digits)
            p_df = p_df.round(round_digits)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(n):
            for j in range(i+1, n):  # Only upper triangle to avoid duplicates
                corr_val = corr_matrix[i, j]
                p_val = p_matrix[i, j]
                
                if (abs(corr_val) >= high_corr_threshold and 
                    p_val <= p_threshold and 
                    not np.isnan(corr_val) and 
                    not np.isnan(p_val)):
                    
                    high_corr_pairs.append({
                        'column1': columns[i],
                        'column2': columns[j],
                        'correlation': corr_val,
                        'p_value': p_val,
                        'abs_correlation': abs(corr_val)
                    })
        
        # Sort by absolute correlation (highest first)
        high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Create summary
        summary = {
            'total_pairs': len(high_corr_pairs),
            'pairs': high_corr_pairs,
            'threshold_used': high_corr_threshold,
            'p_threshold_used': p_threshold
        }
        
        return {
            'correlation_matrix': corr_df,
            'p_value_matrix': p_df,
            'high_corr_summary': summary
        }
    
    else:
        raise TypeError("corr_with_pvalues requires pandas DataFrame")

# Example usage
if __name__ == "__main__":
    # Generate sample data with missing values
    np.random.seed(42)



test = pd.read_csv('https://raw.githubusercontent.com/Plasma-sol/Dirty-data-scan/refs/heads/main/examples/test1.csv?token=GHSAT0AAAAAADFIIAZZITZEW44V7Z2DFWEM2CENWQA')
corr = correlation_matrix(test, method='pearson', round_digits=3)

p_values = corr_with_pvalues(test, round_digits=3)
print("Correlation Matrix:")        
print(p_values[0])  # Correlation matrix
print(p_values[1])

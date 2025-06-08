import numpy as np
import pandas as pd
import re
import os

# Your existing functions (with fixes)
def score_dataset_quality(values, dim1, dim2):
    """
    Score the quality of a dataset based on a vector of values (0-1 range).
    """
    values = np.array(values)
    total_sum = np.sum(values)
    normalized_value = total_sum / (dim1 * dim2)
    
    if normalized_value >= 0.85:
        score = 10
    else:
        score = (normalized_value / 0.85) * 10
    
    return score

def score_duplicate_quality(num_duplicates, dim1, dim2):
    """
    Score the quality of a dataset based on the number of duplicate rows.
    """
    duplicate_ratio = num_duplicates / (dim1 * dim2)
    
    if duplicate_ratio >= 0.25:
        score = 0
    elif duplicate_ratio == 0:
        score = 10
    else:
        score = 10 * (1 - (duplicate_ratio / 0.25))
    
    return score

def score_correlation_redundancy(correlation_df, dim2):
    """
    Score dataset quality based on correlation redundancy between variables.
    """
    high_corr_significant = correlation_df[
        (abs(correlation_df['correlation']) > 0.95) & 
        (correlation_df['p_value'] < 0.05)
    ]
    
    num_redundant_pairs = len(high_corr_significant)
    threshold = 0.1 * dim2
    
    if num_redundant_pairs < threshold:
        score = 10
    else:
        if num_redundant_pairs >= dim2:
            score = 0
        else:
            score = 10 * (1 - (num_redundant_pairs - threshold) / (dim2 - threshold))
            score = max(0, score)
    
    return score, num_redundant_pairs

def score_zscore_quality(avg_zscore):
    """
    Score dataset quality based on average Z-score.
    """
    if avg_zscore <= 3.5:
        score = 10
    elif avg_zscore >= 5:
        score = 0
    else:
        score = 10 * (1 - (avg_zscore - 2.5) / (4.5 - 2.5))
    
    return score

def calculate_overall_dataset_score(values, num_duplicates, correlation_df, avg_zscore, dim1, dim2):
    """
    Calculate overall dataset quality score as a percentage.
    """
    # Calculate individual scores
    quality_score = score_dataset_quality(values, dim1, dim2)
    duplicate_score = score_duplicate_quality(num_duplicates, dim1, dim2)
    correlation_score, num_redundant = score_correlation_redundancy(correlation_df, dim2)  # Fixed: unpack tuple
    zscore_score = score_zscore_quality(avg_zscore)
    
    # Calculate total score and percentage
    total_score = quality_score + duplicate_score + correlation_score + zscore_score
    max_possible_score = 40
    percentage = (total_score / max_possible_score) * 100
    
    # Create detailed results
    results = {
        'individual_scores': {
            'quality_score': quality_score,
            'duplicate_score': duplicate_score,
            'correlation_score': correlation_score,
            'zscore_score': zscore_score
        },
        'total_score': total_score,
        'max_possible_score': max_possible_score,
        'percentage': percentage,
        'num_redundant_pairs': num_redundant
    }
    
    return results

def test_dataset_quality():
    """
    Test the dataset quality scoring functions with the specified files.
    """
    try:
        # File paths (update these to match your actual file locations)
        base_path = r'C:\Users\verdu\OneDrive\Documentos\dirtydata\Dirty-data-scan\examples'
        dimensions_file = os.path.join(base_path, 'dimensions.txt')
        completeness_file = os.path.join(base_path, 'completeness.csv')
        correlation_file = os.path.join(base_path, 'correlation.pkl')
        outliers_file = os.path.join(base_path, 'outliers.pkl')
        duplicates_file = os.path.join(base_path, 'duplicates.csv')
        
        print("Loading data files...")
        
        # Extract dimensions
        with open(dimensions_file, 'r') as file:
            content = file.read()
        
        match = re.search(r'\((\d+),\s*(\d+)\)', content)
        if not match:
            raise ValueError("Could not parse dimensions from file")
        
        dim1, dim2 = int(match.group(1)), int(match.group(2))
        print(f"Dataset dimensions: {dim1} rows × {dim2} columns")
        
        # Read completeness data
        completeness_df = pd.read_csv(completeness_file)
        
        # Read correlation analysis
        correlation_df = pd.read_pickle(correlation_file)
        
        # Read outliers data
        outliers_df = pd.read_pickle(outliers_file)
        
        # Read duplicates data
        duplicates_df = pd.read_csv(duplicates_file)
        
        # Extract required values
        # Assuming completeness_df has a column with completeness ratios (0-1)
        if 'completeness_ratio' in completeness_df.columns:
            values = completeness_df['completeness_ratio'].values
        elif 'completeness' in completeness_df.columns:
            values = completeness_df['completeness'].values
        else:
            # Use the first numeric column
            numeric_cols = completeness_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                values = completeness_df[numeric_cols[0]].values
                print(f"Using column '{numeric_cols[0]}' for completeness values")
            else:
                raise ValueError("No numeric columns found in completeness data")
        
        # Read average Z-score from outliers data
        avg_zscore = outliers_df['avg_of_column_averages']  # Get the single value
        
        # Read duplicates count
        duplicates_file = os.path.join(base_path, 'duplicates.csv')
        duplicates_df = pd.read_csv(duplicates_file)
        num_duplicates = duplicates_df.iloc[0, 0]  # Get the single number from the CSV
        
        print(f"\nInput values:")
        print(f"- Completeness values range: {values.min():.3f} to {values.max():.3f}")
        print(f"- Average Z-score: {avg_zscore:.3f}")
        print(f"- Number of duplicates: {num_duplicates}")
        print(f"- Correlation pairs: {len(correlation_df)}")
        
        # Calculate overall score
        results = calculate_overall_dataset_score(
            values=values,
            num_duplicates=num_duplicates,
            correlation_df=correlation_df,
            avg_zscore=avg_zscore,
            dim1=dim1,
            dim2=dim2
        )
        overall_score = results['total_score']
        
        return overall_score, results
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please check that all required files exist in the specified directory:")
        print("- dimensions.txt")
        print("- completeness.csv") 
        print("- correlation.pkl")
        print("- outliers.pkl")
        print("- duplicates.csv")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    # Run the test
    results = test_dataset_quality()
# # Extract dimensions
#  with open(r'C:\Users\verdu\OneDrive\Documentos\dirtydata\Dirty-data-scan\examples\dimensions.txt', 'r') as file:
#      content = file.read()

#  match = re.search(r'\((\d+),\s*(\d+)\)', content)
#  dim1, dim2 = int(match.group(1)), int(match.group(2))

# # # Read missingness values
#  completeness_df = pd.read_csv(r'C:\Users\verdu\OneDrive\Documentos\dirtydata\Dirty-data-scan\examples\completeness.csv')

# # # Read correlation analysis (dictionary)
    correlation_df = pd.read_pickle(r'C:\Users\verdu\OneDrive\Documentos\dirtydata\Dirty-data-scan\examples\correlation.pkl')

# # # Read outliers (Z-scores)
#  outliers_df = pd.read_pickle(r'C:\Users\verdu\OneDrive\Documentos\dirtydata\Dirty-data-scan\examples\outliers.pkl')
# average_zscore = outliers_df['avg_of_column_averages']



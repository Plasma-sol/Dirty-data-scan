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
   
    analysis = analyze_outliers(df)
    
    # You can access the overall average z-score like this:
    print(f"\nOverall average z-score of all outliers: {analysis['overall_avg_outlier_z_score']:.2f}")
    print(f"Average z-scores per column: {analysis['avg_z_scores_per_column']}")

# test = pd.read_csv('https://raw.githubusercontent.com/Plasma-sol/Dirty-data-scan/refs/heads/main/examples/test1.csv?token=GHSAT0AAAAAADFIIAZZ5ZIS2ESZJ2KGUVDC2CEMYTA')

# analyze_outliers(test, threshold=3)

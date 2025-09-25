import os
import pandas as pd

from utils.registry import load_registry, add_correlation_path_to_registry
from utils.storage import path_validate

# ----------------------- Config -----------------------
correlation_matrices_path = 'data/correlation_matrices/'
path_validate(correlation_matrices_path)

# ----------------------- Correlation Function -----------------------
def correlation_matrix_pearson(df, target_col="target"):
    """
    Computes the Pearson correlation matrix and returns it
    sorted by correlation with the target column.
    """
    correlation_matrix = df.corr(method='pearson')
    sorted_index = correlation_matrix[target_col].sort_values(ascending=False).index
    sorted_corr_matrix = correlation_matrix.loc[sorted_index, sorted_index]
    return sorted_corr_matrix

# ----------------------- Report Generator -----------------------
def generate_report(file_path, prefix):
    df = pd.read_csv(file_path)
    sorted_corr_matrix = correlation_matrix_pearson(df)

    # Save correlation matrix CSV
    corr_csv_path = os.path.join(correlation_matrices_path, f"correlation_{prefix}.csv")
    sorted_corr_matrix.round(2).to_csv(corr_csv_path)

    return sorted_corr_matrix.index, corr_csv_path

# ----------------------- Main Loop -----------------------
registry = load_registry()
target_corr = []      # List of index orderings
dataset_names = []    # Corresponding dataset prefixes

for prefix, info in registry.items():
    if "output_csv_path" not in info:
        print(f"[WARNING] No se encontró 'output_csv_path' para '{prefix}'. Saltando...")
        continue

    file_path = info["output_csv_path"]
    print(f"Generando reporte para '{prefix}' desde '{file_path}'...")

    feature_order, correlation_csv_path = generate_report(file_path, prefix)
    
    # Store feature order and dataset name
    target_corr.append(feature_order)
    dataset_names.append(prefix)

    # Update registry with correlation file path
    add_correlation_path_to_registry(prefix, correlation_csv_path)

# ----------------------- Ranking Matrix -----------------------
# All unique features from all datasets
all_features = sorted(set().union(*target_corr))

# Create a DataFrame for rankings
ranking_df = pd.DataFrame(index=all_features, columns=dataset_names)

# Fill the ranking DataFrame
for col_name, feat_index in zip(dataset_names, target_corr):
    for rank, feature in enumerate(feat_index):
        ranking_df.loc[feature, col_name] = rank + 1  # Ranking starts at 1

# Convert to integers and sort by 'training_data' if exists
ranking_df = ranking_df.astype(int)
if 'training_data' in ranking_df.columns:
    ranking_df = ranking_df.sort_values(by='training_data', ascending=True)

# Save final ranking
ranking_df.to_csv(os.path.join(correlation_matrices_path, "ranking_df.csv"))
print("Correlation and ranking reports generated.")

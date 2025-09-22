import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.registry import load_registry
from utils.storage import path_validate

# ---------------------------- Config ----------------------------
correlation_matrices_path = 'data/correlation_matrices/'
ranking_path = os.path.join(correlation_matrices_path, "ranking_df.csv")

# Validate paths
path_validate(correlation_matrices_path)

# ---------------------------- Load Data ----------------------------
registry = load_registry()

# Load ranking dataframe
ranking_df = pd.read_csv(ranking_path, index_col=0)
ranking_df = ranking_df.sort_values(by="training_data_original", ascending=True)

# ---------------------------- Plot Ranking Heatmap ----------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    ranking_df,
    annot=True,
    cmap='YlGnBu',
    cbar_kws={'label': 'Ranking (1 = High Correlation)'}
)
plt.title("Top Features Ranked by Correlation with the Target Variable")
plt.ylabel("Feature")
plt.xlabel("Data set transformations")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the ranking plot
plt.savefig(os.path.join(correlation_matrices_path, "correlation_matrix_pearson_general_ranking_top.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved general ranking heatmap.")

# ---------------------------- Plot Individual Correlation Matrices ----------------------------
for prefix, info in registry.items():
    corr_path = info.get("correlation_csv_path")
    if not corr_path or not os.path.isfile(corr_path):
        print(f"[WARNING] Skipping '{prefix}' — missing or invalid correlation_csv_path.")
        continue

    corr_df = pd.read_csv(corr_path, index_col=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_df,
        annot=True,
        cmap='coolwarm',
        fmt=".2f"
    )
    plt.title(f"Pearson Correlation Matrix: {prefix}")
    plt.tight_layout()

    output_file = os.path.join(correlation_matrices_path, f"correlation_matrix_pearson_{prefix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation heatmap for: {prefix}")

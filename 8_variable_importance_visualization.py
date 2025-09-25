import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.storage import path_validate
from utils.registry import load_registry

# Paths
importance_path = 'data/variable_importance/'
path_validate(importance_path)

# Load registry
registry = load_registry()

# Filter entries that contain "importance" block
registry_items = [
    (prefix, info["importance"]["features_csv_path"], info["importance"]["metrics_csv_path"])
    for prefix, info in registry.items()
    if "importance" in info
       and os.path.isfile(info["importance"]["features_csv_path"])
       and os.path.isfile(info["importance"]["metrics_csv_path"])
]

# Number of valid models
n_files = len(registry_items)

if n_files == 0:
    print("[ERROR] No valid entries with 'importance' data found in registry.")
    exit()

# Create figure: 3 rows (plots) x n_files (columns)
fig, axes = plt.subplots(3, n_files, figsize=(5 * n_files, 15))
fig.suptitle('Variable Importance and Metrics by Model', fontsize=20)
fig.tight_layout(pad=5.0, rect=[0, 0, 1, 1])

# Ensure axes is always 2D
if n_files == 1:
    axes = axes.reshape(3, 1)

# Loop through registry items
for i, (prefix, features_path, metrics_path) in enumerate(registry_items):
    # Load data
    feature_importance_df = pd.read_csv(features_path)
    metrics_df = pd.read_csv(metrics_path)

    # Plot 1: Top 20 feature importances
    ax1 = axes[0, i]
    top_n = 20
    top_features = feature_importance_df.head(top_n)[::-1]
    ax1.barh(top_features['Feature'], top_features['Importance'])
    ax1.set_title(f'{prefix} - Top 20')
    ax1.set_xlabel('Importance')

    # Plot 2: MAE vs Dimensionality
    ax2 = axes[1, i]
    ax2.plot(metrics_df['num_features'], metrics_df['mae'], marker='o')
    ax2.set_title(f'{prefix} - MAE')
    ax2.set_xlabel('Dimensionality')
    ax2.set_ylabel('MAE')

    # Plot 3: RMSE vs Dimensionality
    ax3 = axes[2, i]
    ax3.plot(metrics_df['num_features'], metrics_df['rmse'], marker='o', color='orange')
    ax3.set_title(f'{prefix} - RMSE')
    ax3.set_xlabel('Dimensionality')
    ax3.set_ylabel('RMSE')

# Save final plot
output_plot_path = os.path.join(importance_path, 'all_plots_grid_from_registry.png')
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[INFO] Plot saved: {output_plot_path}")

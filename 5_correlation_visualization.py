import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.registry import load_registry
from config.paths import CORR_TRAINING_DATA_DIR, CORR_FIGS_DIR

def correlation_figures():
    # ---------------------------- Config ----------------------------
    ranking_path = os.path.join(CORR_TRAINING_DATA_DIR, "ranking_df.csv")

    # ---------------------------- Load Data ----------------------------
    registry = load_registry()

    # Load ranking dataframe
    ranking_df = pd.read_csv(ranking_path, index_col=0)
    ranking_df = ranking_df.sort_values(by=ranking_df.columns[1], ascending=True)

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
    plt.savefig(os.path.join(CORR_FIGS_DIR, "correlation_matrix_pearson_general_ranking_top.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved general ranking heatmap.")

    # ---------------------------- Plot Individual Correlation Matrices ----------------------------
    for prefix, info in registry.items():
        if not info['type'] == 'train':
            continue

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

        output_file = os.path.join(CORR_FIGS_DIR, f"correlation_matrix_pearson_{prefix}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved correlation heatmap for: {prefix}")

if __name__ == '__main__':
    correlation_figures()

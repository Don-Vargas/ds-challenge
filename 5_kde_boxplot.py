import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.storage import path_validate
from utils.registry import load_registry


def kde_boxplot(df, output_prefix):
    kde_boxplot_path = 'data/kde_boxplot/'
    path_validate(kde_boxplot_path)

    features = df.columns
    #num_features = len(features)

    fig = plt.figure(figsize=(14, 19))
    outer = gridspec.GridSpec(5, 5, wspace=0.3, hspace=0.5)

    for i, col in enumerate(features):
        row = i // 2
        col_idx = i % 2
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,  # 2 rows: hist above, boxplot below
            subplot_spec=outer[i],
            height_ratios=[3, 1],  # More space for histogram
            hspace=0.2
        )
        
        # Histogram + KDE
        ax0 = plt.Subplot(fig, inner[0])
        sns.histplot(df[col], kde=True, bins=10, color='skyblue', edgecolor='black', ax=ax0)
        ax0.set_title(col, fontsize=10)
        ax0.set_xlabel("")
        ax0.set_ylabel("Freq")
        fig.add_subplot(ax0)

        # Boxplot with outliers
        ax1 = plt.Subplot(fig, inner[1])
        sns.boxplot(
            x=df[col],
            color='lightgreen',
            ax=ax1,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none')
        )
        ax1.set_xlabel("")
        ax1.set_yticks([])
        fig.add_subplot(ax1)

    plt.suptitle(f"Feature Distributions with KDE + Boxplots {output_prefix}", fontsize=18, y=0.9)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{kde_boxplot_path}kde_boxplot_{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()


# Load the registry (dictionary) from JSON using your utility function
registry = load_registry()

for prefix, info in registry.items():
    output_csv_path = info.get("output_csv_path")
    
    if not output_csv_path or not os.path.isfile(output_csv_path):
        print(f"[WARNING] Skipping '{prefix}' — missing or invalid output_csv_path.")
        continue
    
    # Load transformed data and plot KDE + boxplots
    df = pd.read_csv(output_csv_path)
    kde_boxplot(df, prefix)

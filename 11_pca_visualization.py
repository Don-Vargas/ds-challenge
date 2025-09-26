import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils.registry import load_registry
from config.paths import PCA_FIGS_DIR

# Constants
THRESHOLD = 0.80  # Threshold for cumulative explained variance

# -----------------------------
# Utility Functions
# -----------------------------

def load_pca_model(pickle_path):
    """Load a PCA model from a pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def plot_pca_variance(pca, output_prefix, threshold=THRESHOLD):
    """Generate a single explained variance plot for a PCA model."""
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_selected = np.argmax(cumulative >= threshold) + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker='o', linestyle='--', label='Cumulative variance')
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'{int(threshold * 100)}% variance threshold')
    ax.axvline(x=n_selected, color='g', linestyle='--', label=f'{n_selected} components')
    ax.set_title(f'PCA - {output_prefix}')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_xticks(range(1, len(cumulative) + 1))
    ax.grid(True)
    ax.legend()
    return fig

# -----------------------------
# Main Processing Functions
# -----------------------------

def plot_all_pca_variance_from_registry(threshold=THRESHOLD):
    """
    Load all PCA models from the registry and generate their explained variance plots.
    Returns:
        figs: list of (prefix, matplotlib.figure.Figure)
    """
    registry = load_registry()
    figs = []

    for prefix, info in registry.items():
        if info.get("type") != 'train':
            continue

        input_path = info.get("transformed_data_csv_path")
        if not input_path or not os.path.isfile(input_path):
            print(f"[WARNING] Skipping '{prefix}' — missing or invalid CSV path.")
            continue

        pca_info = info.get("pca", {})
        pca_model_path = pca_info.get("pca_model_pkl_path")

        if not pca_model_path or not os.path.isfile(pca_model_path):
            print(f"[WARNING] Skipping '{prefix}' — PCA model not found.")
            continue

        try:
            pca = load_pca_model(pca_model_path)
            fig = plot_pca_variance(pca, prefix, threshold)
            figs.append((prefix, fig))
        except Exception as e:
            print(f"[ERROR] Failed to plot PCA for '{prefix}': {e}")

    return figs


def save_all_pca_plots_as_grid(figs):
    """Save all PCA plots in a grid layout into a single image file."""
    if not figs:
        print("[INFO] No PCA figures to save.")
        return
    save_path = os.path.join(PCA_FIGS_DIR, 'all_pca_grid.png')

    n = len(figs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig_grid, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    fig_grid.suptitle('PCA - Cumulative Explained Variance per Transformation', fontsize=20)
    fig_grid.tight_layout(pad=5.0, rect=[0, 0, 1, 0.97])

    for ax, (title, fig_item) in zip(axes, figs):
        src_ax = fig_item.axes[0]

        # Copy lines from source figure
        for line in src_ax.lines:
            ax.plot(line.get_xdata(), line.get_ydata(),
                    label=line.get_label(),
                    linestyle=line.get_linestyle(),
                    marker='o' if line.get_marker() else '')

        # Copy threshold and component markers
        for hline in src_ax.lines:
            if hline.get_linestyle() == '--' and hline.get_color() == 'r':
                ax.axhline(y=hline.get_ydata()[0], color='r', linestyle='--')
            if hline.get_linestyle() == '--' and hline.get_color() == 'g':
                ax.axvline(x=hline.get_xdata()[0], color='g', linestyle='--')

        ax.set_title(f'PCA - {title}')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative explained variance')
        ax.grid(True)
        ax.legend()

    # Hide unused subplots
    for ax in axes[len(figs):]:
        ax.axis('off')

    # Save final grid image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] PCA grid plot saved to: {save_path}")

# -----------------------------
# Main Entry Point
# -----------------------------

def create_pca_fig():
    figs = plot_all_pca_variance_from_registry()
    save_all_pca_plots_as_grid(figs)

if __name__ == "__main__":
    create_pca_fig()

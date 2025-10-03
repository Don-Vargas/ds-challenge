import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.storage import path_validate
from config.paths import TEST_PREDICTIONS_PERFORMANCE_FIG_DIR
from utils.registry import load_registry, load_trained_registry


# --- Extract metrics from trained registry ---

def extract_model_metrics_from_registry(trained_registry, metric):
    """
    Extrae las métricas de evaluación (como RMSE o R2), el nombre del modelo,
    el tipo de escalador y la categoría de selección de features desde el registro.
    """
    data = []

    for model_key, model_info in trained_registry.items():
        dataset_name = model_info["dataset_name"]
        model_name = model_info["model_name"]

        # Infer scaler and category from dataset_name
        if "_" in dataset_name:
            parts = dataset_name.split("_")
            scaler = "_".join(parts[:-1])
            category = parts[-1]
        else:
            scaler = dataset_name
            category = "original"

        metric_value = model_info["original_metrics"].get(metric)
        if metric_value is None:
            continue

        data.append({
            "scaler": scaler,
            "category": category,
            "model_name": model_name,
            metric: metric_value
        })

    return pd.DataFrame(data)


# --- Plot differences to reference model ---

def plot_metric_differences(df, metric, reference_model, save_path, grid_title=None,
                            scaler_order=None, transformation_order=None):
    """
    Genera un grid de boxplots con diferencias de la métrica especificada
    contra un modelo de referencia. Se guarda como imagen.
    """
    # Get baseline metric per (scaler, category)
    ref_df = df[df["model_name"] == reference_model][["scaler", "category", metric]]
    ref_df = ref_df.rename(columns={metric: f"{reference_model}_{metric}"})

    # Merge and compute difference
    comp_df = df.merge(ref_df, on=["scaler", "category"], how="left")
    comp_df[f"{metric}_diff"] = comp_df[metric] - comp_df[f"{reference_model}_{metric}"]

    # Plot
    g = sns.catplot(
        data=comp_df,
        x="model_name",
        y=f"{metric}_diff",
        row="scaler",
        col="category",
        kind="box",
        height=3,
        aspect=1.5,
        sharey=False,
        order=sorted(df["model_name"].unique()),
        row_order=scaler_order,
        col_order=transformation_order
    )

    g.set_axis_labels("Modelo", f"Diferencia {metric} vs {reference_model}")
    g.set_titles(row_template="{row_name} scaler", col_template="{col_name} features")

    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    if grid_title:
        g.figure.suptitle(grid_title, fontsize=14)
        g.figure.subplots_adjust(top=1.5)

    g.figure.set_size_inches(14, 20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Main Execution Block ---

if __name__ == "__main__":
    # Load registries
    registry = load_registry()
    trained_registry = load_trained_registry()

    # Create folder if needed
    performance_metrics = TEST_PREDICTIONS_PERFORMANCE_FIG_DIR

    # Define metric to plot
    metric = 'rmse'  # or 'r2'

    # Extract DataFrame of metrics
    df = extract_model_metrics_from_registry(trained_registry, metric)

    # Sort scalers and categories
    scaler_order = sorted(df["scaler"].unique())
    transformation_order = ["important", "pca", "original"]  # Adjust to match your use case

    # Define reference models
    reference_models = ["linear_regression", "ridge"]

    # Generate and save plots
    for ref_model in reference_models:
        save_path = f"{performance_metrics}boxplot_diff_{metric}_{ref_model}.png"
        title = f"Diferencia de {metric} vs {ref_model} (más bajo es mejor)"
        plot_metric_differences(
            df, metric,
            reference_model=ref_model,
            save_path=save_path,
            grid_title=title,
            scaler_order=scaler_order,
            transformation_order=transformation_order
        )

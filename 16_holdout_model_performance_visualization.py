import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.storage import path_validate
from utils.registry import (
    load_registry,
    load_trained_registry,
)

# --- FUNCIÓN 1: Extraer métricas del registry ---

def extract_model_metrics_from_registry(trained_registry):
    """
    Extrae las métricas de RMSE, escalador y tipo de transformación 
    desde un registro de modelos entrenados.
    """
    data = []

    for model_key, model_info in trained_registry.items():
        dataset_name = model_info["dataset_name"]
        model_name = model_info["model_name"]
        scaler = "_".join(dataset_name.split("_")[:-1]) if "_" in dataset_name else dataset_name
        category = dataset_name.split('_')[-1]
        
        rmse = model_info["original_metrics"]["rmse"]

        data.append({
            "scaler": scaler,
            "category": category,
            "model_name": model_name,
            "rmse": rmse
        })

    return pd.DataFrame(data)


# --- FUNCIÓN 2: Plotear diferencias de RMSE vs modelo de referencia ---

def plot_rmse_differences(df, reference_model, save_path, grid_title=None,
                          scaler_order=None, transformation_order=None):
    """
    Genera y guarda un grid de boxplots con las diferencias de RMSE
    comparadas contra un modelo de referencia.
    """
    ref_df = df[df["model_name"] == reference_model][["scaler", "category", "rmse"]]
    ref_df = ref_df.rename(columns={"rmse": f"{reference_model}_rmse"})

    comp_df = df.merge(ref_df, on=["scaler", "category"], how="left")
    comp_df["rmse_diff"] = comp_df["rmse"] - comp_df[f"{reference_model}_rmse"]

    g = sns.catplot(
        data=comp_df,
        x="model_name",
        y="rmse_diff",
        row="scaler",
        col="category",
        kind="box",
        height=4,
        aspect=1.5,
        sharey=False,
        order=sorted(df["model_name"].unique()),
        row_order=scaler_order,
        col_order=transformation_order
    )

    g.set_axis_labels("Modelo", f"Diferencia RMSE vs {reference_model}")
    g.set_titles(row_template="{row_name} scaler", col_template="{col_name} features")

    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    if grid_title:
        g.figure.suptitle(grid_title, fontsize=16)
        g.figure.subplots_adjust(top=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- BLOQUE PRINCIPAL ---

if __name__ == "__main__":
    # Cargar registros
    registry = load_registry()
    trained_registry = load_trained_registry()

    # Crear carpeta si no existe
    performance_metrics = "data/predictions/original_performance_metrics/"
    path_validate(performance_metrics)

    # Extraer DataFrame
    df = extract_model_metrics_from_registry(trained_registry)

    # Definir órdenes de escaladores y transformaciones
    scaler_order = sorted(df["scaler"].unique())
    transformation_order = ["important", "pca", "original"]  # Ajusta si es necesario

    # Modelos de referencia
    reference_models = ["linear_regression", "ridge"]

    # Generar plots
    for ref_model in reference_models:
        save_path = f"{performance_metrics}boxplot_diff_rmse_{ref_model}.png"
        title = f"Diferencia de RMSE vs {ref_model} (más bajo es mejor)"
        plot_rmse_differences(
            df,
            reference_model=ref_model,
            save_path=save_path,
            grid_title=title,
            scaler_order=scaler_order,
            transformation_order=transformation_order
        )

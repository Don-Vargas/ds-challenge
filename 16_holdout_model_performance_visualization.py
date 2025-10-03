import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.paths import TEST_PREDICTIONS_PERFORMANCE_FIG_DIR
from utils.registry import load_registry, load_trained_registry


def extract_model_metrics_from_registry(trained_registry, metric):
    data_list = []
    for key, value in trained_registry.items():
        rmse = value.get('original_metrics', {}).get('rmse', None)
        model_name = value.get('model_name', '')

        parts = model_name.split('_')
        if len(parts) >= 2:
            modelo = parts[0]
            categoria = '_'.join(parts[1:])
        else:
            modelo = model_name
            categoria = ''

        dataset_name = value.get('dataset_name', '')

        data_list.append({
            metric: rmse,
            'modelo': modelo,
            'categoria': categoria,
            'dataset_name': dataset_name
        })

    return pd.DataFrame(data_list)


def prepare_comparison_dataframe(df: pd.DataFrame, ref_model: str, metric: str) -> pd.DataFrame:
    """Agrega columna con diferencia de métrica vs modelo de referencia."""
    ref_df = df[df['modelo'] == ref_model][['categoria', 'dataset_name', metric]]
    ref_df = ref_df.rename(columns={metric: f'{ref_model}_{metric}'})

    df_comp = df.merge(ref_df, on=['categoria', 'dataset_name'], how='left')
    df_comp[f'{metric}_diff'] = df_comp[metric] - df_comp[f'{ref_model}_{metric}']
    return df_comp


def plot_metric_differences(df_comp: pd.DataFrame, metric: str, ref_model: str, save_path: str):
    categorias = ['pca_data', 'most_important', 'transformed_data']
    datasets = ['train', 'train_minmax', 'train_standard', 'train_boxcox', 'train_yeojohnson']
    modelos = ['xgboost', 'linearregression', 'ridge', 'randomforest']

    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(categorias), figsize=(18, 20), sharey=True)

    for i, dataset in enumerate(datasets):
        for j, categoria in enumerate(categorias):
            ax = axes[i, j]
            subset = df_comp[(df_comp['categoria'] == categoria) & (df_comp['dataset_name'] == dataset)]

            sns.boxplot(x='modelo', 
                        y=f'{metric}_diff', 
                        data=subset, 
                        ax=ax, 
                        order=modelos, 
                        palette='viridis')

            ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
            ax.set_title(f'{categoria} | {dataset}')
            ax.set_xlabel('')
            ax.set_ylabel(f'Delta {metric.upper()}' if j == 0 else '')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.suptitle(f'Diferencia de {metric.upper()} respecto a {ref_model}', fontsize=20, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ------------------- MAIN SCRIPT -------------------
def main():
    metric = 'rmse'
    reference_models = ["linearregression", "ridge"]

    # Load registries
    trained_registry = load_trained_registry()
    df = extract_model_metrics_from_registry(trained_registry, metric)

    for ref_model in reference_models:
        df_comp = prepare_comparison_dataframe(df, ref_model, metric)

        diff_save_path = f"{TEST_PREDICTIONS_PERFORMANCE_FIG_DIR}boxplot_{metric}_diff_vs_{ref_model}.png"
        plot_metric_differences(df_comp, metric, ref_model, diff_save_path)


if __name__ == "__main__":
    main()

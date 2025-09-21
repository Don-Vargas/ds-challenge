import seaborn as sns
import matplotlib.pyplot as plt

# 5. Visualización
def visualizar_resultados(results_df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Análisis de Resultados de Modelos", fontsize=18)

    # --- Gráfico 1: Promedio RMSE por modelo ---
    avg_rmse = results_df.groupby('model')['best_rmse'].mean().sort_values()
    sns.barplot(x=avg_rmse.values, y=avg_rmse.index, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Promedio RMSE por Modelo")
    axes[0, 0].set_xlabel("RMSE Promedio")
    axes[0, 0].set_ylabel("Modelo")

    # --- Gráfico 2: Heatmap de RMSE por dataset y modelo ---
    pivot_rmse = results_df.pivot(index='dataset', columns='model', values='best_rmse')
    sns.heatmap(pivot_rmse, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 1])
    axes[0, 1].set_title("RMSE por Dataset y Modelo")

    # --- Gráfico 3: Scatterplot RMSE vs R² ---
    sns.scatterplot(data=results_df, x='best_rmse', y='best_r2', hue='model', style='model', s=100, ax=axes[1, 0])
    axes[1, 0].set_title("RMSE vs R²")
    axes[1, 0].set_xlabel("RMSE")
    axes[1, 0].set_ylabel("R²")

    # --- Gráfico 4: Boxplot de MAE ---
    sns.boxplot(data=results_df, x='model', y='best_mae', ax=axes[1, 1], palette="pastel")
    axes[1, 1].set_title("Distribución de MAE por Modelo")
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].set_xlabel("Modelo")
    axes[1, 1].set_ylabel("MAE")

    # Ajuste final del layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("data/grid_search/visualizaciones_resultados_modelos.png")
    plt.close()

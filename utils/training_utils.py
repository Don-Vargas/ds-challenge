import glob
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from utils.storage import save_pickle

# 1. Métricas
def definir_metricas():
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'rmse': make_scorer(rmse, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }

# 2. Cargar datasets
def obtener_csvs(root_dir, excluded_file):
    csv_files = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
    dataset_dirs = {}

    excluded_file = os.path.normpath(excluded_file)
    excluded_name = os.path.basename(excluded_file)

    for file_path in csv_files:
        file_path_norm = os.path.normpath(file_path)
        file_name = os.path.basename(file_path_norm)

        if file_path_norm == excluded_file or file_name == excluded_name:
            continue

        key = os.path.splitext(file_name)[0]
        dataset_dirs[key] = file_path_norm

    return dataset_dirs

# 3. Modelos y grids
def definir_modelos():
    return {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'model__alpha': [0.01, 0.1, 1, 10, 100],
                'model__solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'saga'],
                'model__fit_intercept': [True, False],
                'model__tol': [1e-3, 1e-4]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 5, 10, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['auto', 'sqrt', 'log2'],
                'model__bootstrap': [True, False]
            }
        },
        'xgboost': {
            'model': XGBRegressor(
                random_state=42,
                verbosity=2,
                tree_method='gpu_hist',
                predictor='gpu_predictor'
            ),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1],
            }
        }
    }

# 4. Entrenamiento
def procesar_dataset(name, path, models, scoring):
    df = pd.read_csv(path)

    if 'target' not in df.columns:
        print(f" 'target' no encontrada en {path}. Saltando archivo.")
        return []

    X = df.drop(columns='target')
    y = df['target']

    resultados = []
    for model_name, config in models.items():
        pipeline = Pipeline([('model', config['model'])])
        grid = GridSearchCV(
            pipeline,
            config['params'],
            scoring=scoring,
            refit='rmse',
            cv=5,
            verbose=2,
            n_jobs=-1,
            return_train_score=False
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                grid.fit(X, y)

            modelo_guardar = grid.best_estimator_
            nombre_archivo = f"data/trained_models/best_model_{name}_{model_name}.pkl"
            save_pickle(modelo_guardar, nombre_archivo)

            cv = grid.cv_results_
            resultados.append({
                'dataset': name,
                'file': os.path.basename(path),
                'model': model_name,
                'best_rmse': -cv['mean_test_rmse'][grid.best_index_] if 'mean_test_rmse' in cv else None,
                'best_mae': -cv['mean_test_mae'][grid.best_index_] if 'mean_test_mae' in cv else None,
                'best_r2': cv['mean_test_r2'][grid.best_index_] if 'mean_test_r2' in cv else None,
                'best_params': grid.best_params_
            })
        except Exception as e:
            print(f"Error al entrenar modelo '{model_name}' con dataset '{name}': {e}")
            continue
    return resultados

# 5. Visualización
def visualizar_resultados(results_df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Análisis de Resultados de Modelos", fontsize=18)

    avg_rmse = results_df.groupby('model')['best_rmse'].mean().sort_values()
    sns.barplot(x=avg_rmse.values, y=avg_rmse.index, ax=axes[0, 0], palette="viridis", hue=avg_rmse.index, legend=False)
    axes[0, 0].set_title("Promedio RMSE por Modelo")

    pivot_rmse = results_df.pivot(index='dataset', columns='model', values='best_rmse')
    sns.heatmap(pivot_rmse, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 1])

    sns.scatterplot(data=results_df, x='best_rmse', y='best_r2', hue='model', style='model', s=100, ax=axes[1, 0])
    axes[1, 0].set_title("RMSE vs R²")

    sns.boxplot(data=results_df, x='model', y='best_mae', ax=axes[1, 1], palette="pastel", hue='model', legend=False)
    axes[1, 1].set_title("Distribución de MAE por Modelo")
    axes[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("data/grid_search/visualizaciones_resultados_modelos.png")
    plt.close()

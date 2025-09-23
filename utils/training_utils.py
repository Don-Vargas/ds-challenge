import os
import warnings
import numpy as np
import pandas as pd
import time

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.storage import save_pickle
from utils.registry import load_registry, add_trained_model_record


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
def obtener_csvs(path):
    return {
        f: os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".csv")
    }

# 3. Modelos y grids
def definir_modelos(seleccionados=None):
    todos_los_modelos = {
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
                'model__min_samples_leaf': [1, 2, 4]
            }
        },
        'xgboost': {
            'model': XGBRegressor(
                random_state=42,
                verbosity=1,
                tree_method='hist'
            ),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1],
            }
        }
    }

    if seleccionados is None:
        return todos_los_modelos
    else:
        return {
            nombre: config
            for nombre, config in todos_los_modelos.items()
            if nombre in seleccionados
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
            verbose=1,
            n_jobs=30,
            return_train_score=False
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                grid.fit(X, y)

            cv = grid.cv_results_  # <- mover acá para que esté disponible

            modelo_guardar = grid.best_estimator_
            nombre_archivo = f"data/trained_models/best_model_{name}_{model_name}.pkl"
            save_pickle(modelo_guardar, nombre_archivo)

            # Actualizar el registry de modelos entrenados
            add_trained_model_record(
                model_name=model_name,
                dataset_name=name,
                model_pickle_path=nombre_archivo,
                best_params=grid.best_params_,
                best_rmse=-cv['mean_test_rmse'][grid.best_index_] if 'mean_test_rmse' in cv else None,
                best_mae=-cv['mean_test_mae'][grid.best_index_] if 'mean_test_mae' in cv else None,
                best_r2=cv['mean_test_r2'][grid.best_index_] if 'mean_test_r2' in cv else None
            )

            resultados.append({
                'dataset': name,
                'file': os.path.basename(path),
                'model': model_name,
                'best_rmse': -cv['mean_test_rmse'][grid.best_index_] if 'mean_test_rmse' in cv else None,
                'best_mae': -cv['mean_test_mae'][grid.best_index_] if 'mean_test_mae' in cv else None,
                'best_r2': cv['mean_test_r2'][grid.best_index_] if 'mean_test_r2' in cv else None,
                'best_params': grid.best_params_
            })
            time.sleep(1)
        except Exception as e:
            print(f"Error al entrenar modelo '{model_name}' con dataset '{name}': {e}")
            continue

    return resultados

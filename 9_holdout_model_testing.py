import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from utils.storage import path_validate


# Carpeta de modelos y datasets
MODELS_FOLDER = "data/trained_models/"
HOLDOUT_FOLDER = "data/data_for_models/split_datasets/holdout/"

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def evaluar_modelos(models_folder, holdout_folder):
    resultados = []

    # Listar todos los modelos .pkl
    for model_file in os.listdir(models_folder):
        if not model_file.endswith(".pkl"):
            continue

        model_path = os.path.join(models_folder, model_file)

        # Extraer nombre de dataset y modelo desde el filename
        # Formato esperado: best_model_<dataset>_<model>.pkl
        # 1. Quitar prefijo y sufijo
        name_ext = model_file.replace("best_model_", "").replace(".pkl","")

        # 2. Separar dataset y modelo usando '.csv' como referencia
        dataset_name = name_ext.split(".csv")[0] + ".csv"   # everything hasta '.csv'
        model_name = name_ext.split(".csv")[1].lstrip("_")  # lo que queda después del '.csv' (quitando '_')



        # Cargar modelo
        model = load_model(model_path)

        # Buscar el dataset holdout correspondiente
        dataset_path = os.path.join(holdout_folder, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Holdout no encontrado: {dataset_name}")
            continue

        # Cargar datos
        df = pd.read_csv(dataset_path)
        if 'target' not in df.columns:
            print(f"'target' no encontrada en {dataset_name}")
            continue

        X_holdout = df.drop(columns='target')
        y_holdout = df['target']

        # Predecir
        y_pred = model.predict(X_holdout)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_holdout, y_pred))
        mae = mean_absolute_error(y_holdout, y_pred)
        r2 = r2_score(y_holdout, y_pred)

        # Obtener hiperparámetros si existen
        try:
            params = model.get_params()
        except:
            params = {}

        resultados.append({
            "dataset": dataset_name,
            "file": dataset_name,
            "model": model_name,
            "best_rmse": rmse,
            "best_mae": mae,
            "best_r2": r2,
            "best_params": params
        })

    return pd.DataFrame(resultados)

# Ejecutar evaluación
df_resultados = evaluar_modelos(MODELS_FOLDER, HOLDOUT_FOLDER)

# Define conditions based on the 'dataset' column
conditions = [
    df_resultados['dataset'].str.contains('pca', case=False),
    df_resultados['dataset'].str.contains('important', case=False)
]
# Define the corresponding choices
choices = ['pca', 'important']
# Create the new column, defaulting to 'None' if no condition matches
df_resultados['category'] = np.select(conditions, choices, default='plain')

# Guardar resultados
hold_out_eval_path = 'data/hold_out_evaluation/'
path_validate(hold_out_eval_path)
df_resultados.to_csv(f"{hold_out_eval_path}evaluacion_holdout.csv", index=False)

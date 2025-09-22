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

def revert_transformations(X_holdout, y_pred, y_holdout, dataset_name):
    """
    Invierte las transformaciones (PCA y Scaler) para obtener y_pred e y_holdout en su escala original.
    """
    # Quitar .csv
    base_name = dataset_name.replace('.csv', '')

    # Eliminar prefijos como 'important_' o 'pca_' si existen
    for prefix in ['important_', 'pca_']:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]

    # Paths
    scaler_path = f"data/scaler_models/scaler_{base_name}.pkl"
    pca_path = f"data/pca_models/pca_model_{base_name}.pkl"

    # Flags
    has_pca = os.path.exists(pca_path)
    has_scaler = os.path.exists(scaler_path)

    if not has_scaler:
        raise FileNotFoundError(f"No se encontró scaler para {base_name}")

    # Cargar scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Cargar pca si existe
    if has_pca:
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)

        # Revertir PCA solo en features
        X_recovered = pca.inverse_transform(X_holdout)
    else:
        X_recovered = X_holdout

    # Reinsertar y_pred y y_holdout en la última columna de X para poder invertir con scaler
    X_pred_full = np.hstack([X_recovered, y_pred.reshape(-1, 1)])
    X_true_full = np.hstack([X_recovered, y_holdout.values.reshape(-1, 1)])

    # Inversión de scaler (este espera features + target)
    X_pred_orig = scaler.inverse_transform(X_pred_full)
    X_true_orig = scaler.inverse_transform(X_true_full)

    # Extraer la última columna como target
    y_pred_orig = X_pred_orig[:, -1]
    y_true_orig = X_true_orig[:, -1]

    return y_pred_orig, y_true_orig

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

        # SKIP si el dataset viene de la carpeta de "most_important_features"
        if 'important_training_data.csv' in dataset_name:
            print(f"Skip explícito para: {dataset_name}")
            continue
        
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

        try:
            y_pred_orig, y_true_orig = revert_transformations(X_holdout.values, y_pred, y_holdout, dataset_name)
        except Exception as e:
            print(f"Error revirtiendo transformaciones para {dataset_name}: {e}")
            continue

        # Calcular métricas en escala original
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        r2 = r2_score(y_true_orig, y_pred_orig)

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

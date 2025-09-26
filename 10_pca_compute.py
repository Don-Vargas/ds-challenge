import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.storage import save_pickle, load_pickle
from utils.registry import load_registry, add_pca_data_path_to_registry
from config.paths import (
    TRAINING_DATA_PCA_DIR, TEST_DATA_PCA_DIR, BLIND_DATA_PCA_DIR,
    TRAINING_MODEL_PCA_DIR
)
#NOTE: actualizar codigo para que entrene train pca y guarde sus modelos y a partir de esos modelos .fit test y blind.

VARIANCE_THRESHOLD = 0.80

def apply_pca(X):
    """
    Ajusta PCA a los datos X y selecciona las componentes necesarias
    para explicar al menos 'variance_threshold' de la varianza.

    Retorna un diccionario con el modelo PCA y la info necesaria
    para aplicarlo a nuevos datos.
    """
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_selected = np.argmax(cumulative >= VARIANCE_THRESHOLD) + 1
    
    # Selecciona solo las primeras componentes
    X_pca_sel = X_pca_full[:, :n_selected]
    
    return {
        'pca_model': pca,
        'n_selected': n_selected,
        'X_pca': X_pca_sel
    }

def get_pca_object(prefix):
    """
    Carga el modelo PCA correspondiente al prefijo de entrenamiento.
    Si el prefijo es 'test' o 'blind' sin sufijo, se asume que corresponde a 'train'.
    """
    if prefix == "test":
        train_prefix = "train"
    elif prefix == "blind":
        train_prefix = "train"
    elif prefix.startswith("test_"):
        train_prefix = prefix.replace("test_", "train_")
    elif prefix.startswith("blind_"):
        train_prefix = prefix.replace("blind_", "train_")
    else:
        raise ValueError(f"[ERROR] No matching train model for prefix '{prefix}'.")

    model_path = os.path.join(TRAINING_MODEL_PCA_DIR, f'pca_model_{train_prefix}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] PCA model not found for prefix: '{train_prefix}'")

    return model_path


def process_and_save(prefix, info, output_data_folder, output_model_folder, tipo):
    input_path = info.get("transformed_data_csv_path")
    df = pd.read_csv(input_path)
    if df is None:
        return

    try:
        if tipo != 'blind':
            y = df['target']
            X = df.drop(columns=['target'])
        else:
            X = df

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if tipo == 'train':
            pca_object = apply_pca(X_arr)
        else:
            training_pca_model_path = get_pca_object(prefix)
            saved_pca = load_pickle(training_pca_model_path)
            pca_model = saved_pca['pca_model']
            n_selected = saved_pca['n_selected']
            
            X_pca_full = pca_model.transform(X_arr)
            X_pca_sel = X_pca_full[:, :n_selected]
            
            pca_object = {
                'pca_model': pca_model,
                'n_selected': n_selected,
                'X_pca': X_pca_sel
            }
        col_names = [f'PC_{i+1}' for i in range(pca_object['n_selected'])]
        df_pca = pd.DataFrame(pca_object['X_pca'], columns=col_names)

        if tipo != 'blind':
            df_pca['target'] = y

        output_csv_path = os.path.join(output_data_folder, f'pca_{prefix}.csv')
        df_pca.to_csv(output_csv_path, index=False)

        if tipo == 'train':
            pca_model_path = os.path.join(output_model_folder, f'pca_model_{prefix}.pkl')
            save_pickle(pca_object, pca_model_path)
            add_pca_data_path_to_registry(prefix, [output_csv_path, pca_model_path])
        else:
            add_pca_data_path_to_registry(prefix, [output_csv_path, training_pca_model_path])

    except Exception as e:
        print(f"[ERROR] Failed processing '{prefix}': {e}")

def pca():
    registry = load_registry()
    for prefix, info in registry.items():
        if info.get("type") == 'train':
            process_and_save(prefix, info,
                         output_data_folder=TRAINING_DATA_PCA_DIR,
                         output_model_folder=TRAINING_MODEL_PCA_DIR,
                         tipo=info.get("type"))

    for prefix, info in registry.items():
        if info.get("type") == 'test':
            process_and_save(prefix, info,
                         output_data_folder=TEST_DATA_PCA_DIR,
                         output_model_folder=TRAINING_MODEL_PCA_DIR,
                         tipo=info.get("type"))

    for prefix, info in registry.items():
        if info.get("type") == 'blind':
            process_and_save(prefix, info,
                         output_data_folder=BLIND_DATA_PCA_DIR,
                         output_model_folder=TRAINING_MODEL_PCA_DIR,
                         tipo=info.get("type"))

if __name__ == "__main__":
    pca()

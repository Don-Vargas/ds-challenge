import os
import pickle

TRAINING_DATA = 'data/data_for_models/original_data/training_data.csv'
BLIND_DATA = 'data/data_for_models/original_data/blind_test_data.csv'

def path_validate(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_pickle(obj, filepath):
    """
    Guarda un objeto en formato pickle en la ruta indicada.
    Si la carpeta no existe, la crea automáticamente.

    Args:
        obj: Objeto Python a guardar.
        filepath (str): Ruta completa del archivo destino (ej. 'data/scaler/scaler.pkl').
    """
    # Crear carpeta si no existe
    path_validate(filepath)

    # Guardar objeto
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    print(f"Objeto guardado en {filepath}")

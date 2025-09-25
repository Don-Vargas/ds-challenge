import os
import pickle

def path_validate(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def path_validate(path):
    """
    Si `path` es una carpeta, la crea si no existe.
    Si `path` es un archivo, crea la carpeta contenedora.
    """
    directory = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

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

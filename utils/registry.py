import json
import os

from utils.storage import path_validate

REGISTRY_FILE = 'registry/model_registry.json'
path_validate(REGISTRY_FILE)

def load_registry(file_path=REGISTRY_FILE):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as f:
        return json.load(f)

def save_registry(registry, file_path=REGISTRY_FILE):
    with open(file_path, 'w') as f:
        json.dump(registry, f, indent=4)

def add_transformation_record(output_prefix, 
                              transformer, 
                              columns, 
                              output_csv_path, 
                              scaler_pickle_path=None):
    """
    Agrega un registro de transformación al archivo JSON.
    """
    registry = load_registry()

    transformer_info = {
        'transformation':{
            'transformer_type': type(transformer).__name__ if transformer else 'None',
            'params': transformer.get_params() if transformer else {}
        },
        'columns': columns,
        'scaler_pickle_path': scaler_pickle_path,
        'output_csv_path': output_csv_path
    }

    registry[output_prefix] = transformer_info
    save_registry(registry)

def add_eda_path_to_registry(output_prefix, eda_report_path, file_path=REGISTRY_FILE):
    """
    Agrega o actualiza la ruta del reporte EDA en el registro existente para un prefix dado.
    """
    registry = load_registry(file_path)

    if output_prefix not in registry:
        print(f"[WARNING] La llave '{output_prefix}' no existe en el registro. No se actualizó nada.")
        return

    registry[output_prefix]['eda_report_path'] = eda_report_path
    save_registry(registry, file_path)
    print(f"[INFO] Ruta EDA agregada a '{output_prefix}': {eda_report_path}")

def add_correlation_path_to_registry(output_prefix, correlation_csv_path, file_path=REGISTRY_FILE):
    """
    Agrega o actualiza la ruta del reporte de correlacion en el registro existente para un prefix dado.
    """
    registry = load_registry(file_path)

    if output_prefix not in registry:
        print(f"[WARNING] La llave '{output_prefix}' no existe en el registro. No se actualizó nada.")
        return

    registry[output_prefix]['correlation_csv_path'] = correlation_csv_path
    save_registry(registry, file_path)
    print(f"[INFO] Ruta EDA agregada a '{output_prefix}': {correlation_csv_path}")

def add_importance_path_to_registry(output_prefix, importance_paths, file_path=REGISTRY_FILE):
    """
    Agrega o actualiza las rutas del reporte de importancia de variables en el registro existente para un prefix dado.
    """
    registry = load_registry(file_path)

    if output_prefix not in registry:
        print(f"[WARNING] La llave '{output_prefix}' no existe en el registro. No se actualizó nada.")
        return

    # Crear la clave 'importance' si no existe
    if 'importance' not in registry[output_prefix]:
        registry[output_prefix]['importance'] = {}

    registry[output_prefix]['importance']['features_csv_path'] = importance_paths[0]
    registry[output_prefix]['importance']['metrics_csv_path'] = importance_paths[1]

    save_registry(registry, file_path)
    print(f"[INFO] Rutas de importancia agregadas a '{output_prefix}': {importance_paths}")

def add_most_important_data_path_to_registry(output_prefix, most_important_csv_path, file_path=REGISTRY_FILE):
    """
    Agrega o actualiza la ruta de variables mas importantes en el registro existente para un prefix dado.
    """
    registry = load_registry(file_path)

    if output_prefix not in registry:
        print(f"[WARNING] La llave '{output_prefix}' no existe en el registro. No se actualizó nada.")
        return

    registry[output_prefix]['most_important_csv_path'] = most_important_csv_path
    save_registry(registry, file_path)
    print(f"[INFO] Ruta de variables mas importantes agregada a '{output_prefix}': {most_important_csv_path}")

def add_pca_data_path_to_registry(output_prefix, pcs_csv_path, file_path=REGISTRY_FILE):
    """
    Agrega o actualiza la ruta de PCA en el registro existente para un prefix dado.
    """
    registry = load_registry(file_path)

    if output_prefix not in registry:
        print(f"[WARNING] La llave '{output_prefix}' no existe en el registro. No se actualizó nada.")
        return
    # Crear la clave 'importance' si no existe
    if 'pca' not in registry[output_prefix]:
        registry[output_prefix]['pca'] = {}

    registry[output_prefix]['pca']['pca_data_csv_path'] = pcs_csv_path[0]
    registry[output_prefix]['pca']['pca_model_pkl_path'] = pcs_csv_path[1]
    save_registry(registry, file_path)
    print(f"[INFO] Ruta de PCA agregada a '{output_prefix}': {pcs_csv_path}")

def add_split_data_path_to_registry(output_prefix, train_hold_csv_path, file_path=REGISTRY_FILE):
    """
    Agrega o actualiza la ruta de los data sets train y hold-out en el registro existente para un prefix dado.
    """
    registry = load_registry(file_path)

    if output_prefix not in registry:
        print(f"[WARNING] La llave '{output_prefix}' no existe en el registro. No se actualizó nada.")
        return
    # Crear la clave 'importance' si no existe
    if 'train_hold' not in registry[output_prefix]:
        registry[output_prefix]['train_hold'] = {}

    registry[output_prefix]['train_hold']['train_data_csv_path'] = train_hold_csv_path[0]
    registry[output_prefix]['train_hold']['hold_out_data_csv_path'] = train_hold_csv_path[1]
    save_registry(registry, file_path)
    print(f"[INFO] Ruta de los data sets train y hold-out agregada a '{output_prefix}': {train_hold_csv_path}")



######################################

TRAINED_REGISTRY_FILE = 'registry/trained_model_registry.json'
path_validate(TRAINED_REGISTRY_FILE)

def load_trained_registry(file_path=TRAINED_REGISTRY_FILE):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as f:
        return json.load(f)

def save_trained_registry(registry, file_path=TRAINED_REGISTRY_FILE):
    with open(file_path, 'w') as f:
        json.dump(registry, f, indent=4)

def add_trained_model_record(model_name,
                             dataset_name,
                             model_pickle_path,
                             best_params=None,
                             best_rmse=None,
                             best_mae=None,
                             best_r2=None,
                             model_registry_path=REGISTRY_FILE,
                             trained_registry_path=TRAINED_REGISTRY_FILE):
    model_registry = load_registry(model_registry_path)
    trained_registry = load_trained_registry(trained_registry_path)

    dataset_info = model_registry.get(dataset_name, {})

    transformer_info = {
        'scaler_pickle_path': dataset_info.get('scaler_pickle_path'),
        'pca_model_pkl_path': dataset_info.get('pca', {}).get('pca_model_pkl_path')
    }

    key = f"{model_name}_{dataset_name}"
    trained_registry[key] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "transformer_info": transformer_info,
        "model_pickle_path": model_pickle_path,
        "training_info": {
            "best_params": best_params,
            "best_rmse": best_rmse,
            "best_mae": best_mae,
            "best_r2": best_r2
        }
    }

    save_trained_registry(trained_registry, trained_registry_path)
    print(f"[INFO] Registro de modelo entrenado '{key}' actualizado.")

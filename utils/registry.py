import json
import os

from utils.storage import path_validate
from config.paths import REGISTRY_FILE, TRAINED_REGISTRY_FILE

# ---------------------- #
#    Rutas y Validación  #
# ---------------------- #

# Funciones Base Reutilizables
def _load_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as f:
        return json.load(f)

def _save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def _update_registry_entry(file_path, key, update_dict):
    registry = _load_json(file_path)
    if key not in registry:
        print(f"[WARNING] La llave '{key}' no existe en el registro. No se actualizó nada.")
        return

    registry[key].update(update_dict)
    _save_json(registry, file_path)
    print(f"[INFO] Registro actualizado para '{key}' con: {list(update_dict.keys())}")

# Registro General

def load_registry():
    return _load_json(REGISTRY_FILE)

def save_registry(registry):
    _save_json(registry, REGISTRY_FILE)

def add_transformation_record(output_prefix, transformer, transformed_data_csv_path, scaler_pickle_path=None, blind=False):
    registry = load_registry()

    transformer_info = {
        'transformation': {
            'transformer_type': type(transformer).__name__ if transformer else 'None',
            'params': transformer.get_params() if transformer else {}
        },
        'scaler_pickle_path': scaler_pickle_path,
        'transformed_data_csv_path': transformed_data_csv_path,
        'blind': blind
    }

    registry[output_prefix] = transformer_info
    save_registry(registry)
    print(f"[INFO] Transformación registrada para '{output_prefix}'")


def add_eda_path_to_registry(output_prefix, eda_report_path):
    _update_registry_entry(REGISTRY_FILE, output_prefix, {
        'eda_report_path': eda_report_path
    })

def add_correlation_path_to_registry(output_prefix, correlation_csv_path):
    _update_registry_entry(REGISTRY_FILE, output_prefix, {
        'correlation_csv_path': correlation_csv_path
    })

def add_importance_path_to_registry(output_prefix, importance_paths):
    _update_registry_entry(REGISTRY_FILE, output_prefix, {
        'importance': {
            'features_csv_path': importance_paths[0],
            'metrics_csv_path': importance_paths[1]
        }
    })

def add_most_important_data_path_to_registry(output_prefix, most_important_csv_path):
    _update_registry_entry(REGISTRY_FILE, output_prefix, {
        'most_important_csv_path': most_important_csv_path
    })

def add_pca_data_path_to_registry(output_prefix, pcs_csv_path):
    _update_registry_entry(REGISTRY_FILE, output_prefix, {
        'pca': {
            'pca_data_csv_path': pcs_csv_path[0],
            'pca_model_pkl_path': pcs_csv_path[1]
        }
    })

# Registro de Modelos Entrenados

def load_trained_registry():
    return _load_json(TRAINED_REGISTRY_FILE)

def save_trained_registry(registry):
    _save_json(registry, TRAINED_REGISTRY_FILE)

def add_trained_model_record(model_name,
                             dataset_name,
                             model_pickle_path,
                             best_params=None,
                             best_rmse=None,
                             best_mae=None,
                             best_r2=None):
    registry = load_registry()
    trained_registry = load_trained_registry()

    dataset_info = registry.get(dataset_name, {})
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

    save_trained_registry(trained_registry)
    print(f"[INFO] Registro de modelo entrenado '{key}' actualizado.")


def add_predictions_data_path_to_registry(model_key, predictions_csv_path):
    _update_registry_entry(TRAINED_REGISTRY_FILE, model_key, {
        'predictions_transformed_scale_csv_path': predictions_csv_path
    })

def add_predictions_original_scale_data_path_to_registry(model_key, predictions_csv_path):
    _update_registry_entry(TRAINED_REGISTRY_FILE, model_key, {
        'predictions_original_scale_csv_path': predictions_csv_path
    })

def add_original_metrics_path_to_registry(model_key, metrics):
    _update_registry_entry(TRAINED_REGISTRY_FILE, model_key, {
        'original_metrics': {
            'rmse': metrics[0],
            'mae': metrics[1],
            'r2': metrics[2]
        }
    })

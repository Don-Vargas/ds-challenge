import pandas as pd
import numpy as np
import os
import joblib

from utils.storage import path_validate
from utils.registry import (
    load_registry,
    load_trained_registry,
    add_predictions_original_scale_data_path_to_registry
)


def restaurar_predicciones_modelo(model_info, dataset_info, output_dir):
    """
    Restaura las predicciones de un modelo específico a escala original.

    Parámetros:
        model_info (dict): Información del modelo entrenado desde el registry.
        dataset_info (dict): Información del dataset transformado desde el registry.
        output_dir (str): Carpeta donde guardar las predicciones desescaladas.
    """
    predictions_path = model_info.get("predictions_transformed_scale_csv_path")
    dataset_name = model_info.get("dataset_name")
    model_name = model_info.get("model_name")

    if not predictions_path or not os.path.exists(predictions_path):
        print(f"Predicciones no encontradas para modelo '{model_name}'. Se omite.")
        return

    blind = "blind" in dataset_name.lower() or "blind" in predictions_path.lower()
    print(f"Procesando modelo '{model_name}' para dataset '{dataset_name}' (blind={blind})")

    scaler_path = dataset_info.get("scaler_pickle_path")

    df_preds = pd.read_csv(predictions_path)
    preds = df_preds.values

    if scaler_path:
        if isinstance(scaler_path, dict) and "y" in scaler_path:
            y_scaler_path = scaler_path["y"]
        elif isinstance(scaler_path, str):
            y_scaler_path = scaler_path
        else:
            print("Formato de scaler_path no reconocido. Se omite desescalado.")
            y_scaler_path = None

        if y_scaler_path and os.path.exists(y_scaler_path):
            print(f"  Revirtiendo escala con scaler: {y_scaler_path}")
            scaler_y = joblib.load(y_scaler_path)

            try:
                if blind:
                    y_pred_original = scaler_y.inverse_transform(preds)
                    preds = y_pred_original
                else:
                    y_true_original = scaler_y.inverse_transform(preds[:, [0]])
                    y_pred_original = scaler_y.inverse_transform(preds[:, [1]])
                    preds = np.hstack([y_true_original, y_pred_original])
            except Exception as e:
                print(f"Error al revertir la escala: {e}")
        else:
            print("Scaler no encontrado o inválido. Se omite desescalado.")
    else:
        print("No se aplicó escalado. Se mantienen valores originales.")

    if blind:
        df_preds_original = pd.DataFrame(preds, columns=["y_pred"])
    else:
        if preds.shape[1] == 2:
            df_preds_original = pd.DataFrame(preds, columns=["y_true", "y_pred"])
        else:
            df_preds_original = pd.DataFrame(preds)

    predictions_csv_filename = f"predictions_{dataset_name}_{model_name}.csv"
    predictions_csv_path = os.path.join(output_dir, predictions_csv_filename)
    df_preds_original.to_csv(predictions_csv_path, index=False)

    add_predictions_original_scale_data_path_to_registry(
        model_name=f'{model_name}_{dataset_name}',
        predictions_csv_path=predictions_csv_path
    )

    print(f"Predicciones guardadas en escala original: {predictions_csv_path}")

def restaurar_predicciones_a_escala_original(
    output_dir: str = "data/predictions/original_scale/",
    trained_models_registry: dict = None,
    datasets_registry: dict = None
):
    """
    Itera sobre todos los modelos entrenados y restaura sus predicciones a escala original.

    Parámetros:
        output_dir (str): Carpeta donde se guardarán las predicciones desescaladas.
        trained_models_registry (dict, opcional): Registro de modelos entrenados. Se carga automáticamente si no se pasa.
        datasets_registry (dict, opcional): Registro de datasets transformados. Se carga automáticamente si no se pasa.
    """
    path_validate(output_dir)

    if trained_models_registry is None:
        trained_models_registry = load_trained_registry()
    if datasets_registry is None:
        datasets_registry = load_registry()

    for model_key, model_info in trained_models_registry.items():
        dataset_name = model_info.get("dataset_name")
        base_dataset_name = "_".join(dataset_name.split("_")[:-1]) if "_" in dataset_name else dataset_name

        if base_dataset_name not in datasets_registry:
            print(f"No se encontró el dataset '{base_dataset_name}' en el registry. Se omite.")
            continue

        dataset_info = datasets_registry[base_dataset_name]
        restaurar_predicciones_modelo(model_info, dataset_info, output_dir)

if __name__ == "__main__":
    restaurar_predicciones_a_escala_original()

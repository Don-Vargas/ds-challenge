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

# Cargar registros
registry = load_registry()
trained_registry = load_trained_registry()

# Crear carpeta destino
original_preds_dir = "data/predictions/original_scale/"
path_validate(original_preds_dir)

# Iterar sobre modelos entrenados
for model_key, model_info in trained_registry.items():
    predictions_path = model_info["predictions_transformed_scale_csv_path"]
    dataset_name = model_info["dataset_name"]
    model_name = model_info["model_name"]
    filename = os.path.basename(predictions_path)

    # Obtener nombre base del dataset
    base_dataset_name = "_".join(dataset_name.split("_")[:-1]) if "_" in dataset_name else dataset_name
    print(f"  Dataset base: {base_dataset_name}")

    if base_dataset_name not in registry:
        print(f"  No se encontró el dataset '{base_dataset_name}' en el registry. Se omite.")
        continue

    dataset_info = registry[base_dataset_name]
    scaler_path = dataset_info.get("scaler_pickle_path")
    pca_path = dataset_info.get("pca", {}).get("pca_model_pkl_path")

    # Leer predicciones
    df_preds = pd.read_csv(predictions_path)
    preds = df_preds.values

    # Revertir escalado (solo para variable Y)
    if scaler_path:
        if isinstance(scaler_path, dict) and "y" in scaler_path:
            y_scaler_path = scaler_path["y"]
            print(f"  Revirtiendo transformación del scaler (y) desde: {y_scaler_path}")
            scaler_y = joblib.load(y_scaler_path)
            y_true_original = scaler_y.inverse_transform(preds[:, [0]])
            y_pred_original = scaler_y.inverse_transform(preds[:, [1]])
            preds = np.hstack([y_true_original, y_pred_original])
        elif isinstance(scaler_path, str):
            print(f"  Revirtiendo transformación del scaler desde: {scaler_path}")
            scaler_y = joblib.load(y_scaler_path)
            y_true_original = scaler_y.inverse_transform(preds[:, [0]])
            y_pred_original = scaler_y.inverse_transform(preds[:, [1]])
            preds = np.hstack([y_true_original, y_pred_original])
        else:
            print(f"  Formato de scaler_path no reconocido: {scaler_path}")
    else:
        print(f"  No se aplicó ningún escalado. Se mantiene escala original.")

    # Asignar nombres de columnas si hay dos columnas (y_true, y_pred)
    if preds.shape[1] == 2:
        df_preds_original = pd.DataFrame(preds, columns=["y_true", "y_pred"])
    else:
        df_preds_original = pd.DataFrame(preds)
        print("  No se asignaron nombres de columnas: se esperaban 2 columnas (y_true, y_pred).")

    # Generate filename
    predictions_csv_filename = f"predictions_{dataset_name}_{model_name}.csv"
    predictions_csv_path = os.path.join(original_preds_dir, predictions_csv_filename)

    # Save predictions
    df_preds_original.to_csv(predictions_csv_path, index=False)

    # Actualizar registro
    add_predictions_original_scale_data_path_to_registry(
        model_name=f'{model_name}_{dataset_name}',
        predictions_csv_path=predictions_csv_path
    )

    print(f"  Predicciones en escala original guardadas en: {predictions_csv_path}")

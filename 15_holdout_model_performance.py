import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from utils.registry import (
    load_registry,
    load_trained_registry,
    add_original_metrics_path_to_registry
)

# Cargar registros
registry = load_registry()
trained_registry = load_trained_registry()

# Crear carpeta destino si no existe
predictions = "data/predictions/original_scale/"

for model_key, model_info in trained_registry.items():
    predictions_path = model_info["predictions_transformed_scale_csv_path"]
    dataset_name = model_info["dataset_name"]
    model_name = model_info["model_name"]
    filename = os.path.basename(predictions_path)

    # Obtener nombre base del dataset
    base_dataset_name = f'{model_name}_{dataset_name}'
    print(f"  base_dataset_name: {base_dataset_name}")
    print(f"  Dataset base: {dataset_name}")
    
    # Construir ruta al archivo de predicciones en escala original
    # Asumiendo que existe un archivo para cada modelo con esa convención:
    pred_file = os.path.join(predictions, f"predictions_{dataset_name}_{model_name}.csv")
    
    if not os.path.exists(pred_file):
        print(f"Archivo no encontrado para {model_name}: {pred_file}")
        continue
    
    df = pd.read_csv(pred_file)
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # Guardar las métricas en el registro (puedes adaptar esta función)
    add_original_metrics_path_to_registry(base_dataset_name, [rmse, mae, r2])

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.storage import path_validate
from utils.registry import (
    load_registry,
    load_trained_registry,
    add_original_metrics_path_to_registry
)

# Cargar registros
registry = load_registry()
trained_registry = load_trained_registry()

# Filtrar modelos xgboost
xgboost_models = {k: v for k, v in trained_registry.items() if v.get("model_name") == "xgboost"}

# Crear lista para almacenar resultados
results = []

for model_key, model_info in xgboost_models.items():
    rmse = model_info.get("original_metrics", {}).get("rmse", None)
    results.append({
        "model_key": model_key,
        "dataset_name": model_info.get("dataset_name"),
        "rmse": rmse
    })

# Convertir a DataFrame
df_rmse = pd.DataFrame(results)

# Ordenar por RMSE ascendente (mejor primero)
df_rmse_sorted = df_rmse.sort_values(by="rmse")

print(df_rmse_sorted)
print('-------------------------------------')
for model_key, model_info in trained_registry.items():
    rmse = model_info.get("original_metrics", {}).get("rmse", None)
    results.append({
        "model_key": model_key,
        "model_name": model_info.get("model_name"),
        "dataset_name": model_info.get("dataset_name"),
        "rmse": rmse
    })

# Convertir a DataFrame
df_rmse = pd.DataFrame(results)

# Eliminar filas donde rmse es None (si hay)
df_rmse = df_rmse.dropna(subset=["rmse"])

# Ordenar por RMSE ascendente y obtener top 5
top5_models = df_rmse.sort_values(by="rmse").head(5)

print(top5_models)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.registry import (
    load_registry,
    load_trained_registry
)

# === CONFIGURATION ===
selected_metric = "rmse"

# Define sort direction
metric_sorting = {
    "rmse": {"ascending": True},
    "mae": {"ascending": True},
    "r2": {"ascending": False}
}
# ======================

# Load registries
registry = load_registry()
trained_registry = load_trained_registry()

# Filter XGBoost models by partial match
xgboost_models = {
    k: v for k, v in trained_registry.items()
    if "xgboost" in v.get("model_name", "")
}

# Collect XGBoost results
results = []

for model_key, model_info in xgboost_models.items():
    metric_value = model_info.get("original_metrics", {}).get(selected_metric)
    results.append({
        "model_key": model_key,
        "dataset_name": model_info.get("dataset_name"),
        selected_metric: metric_value
    })

# Create DataFrame and clean
df_metric = pd.DataFrame(results)

if selected_metric not in df_metric.columns:
    print(f"No XGBoost models found with metric '{selected_metric}'.")
else:
    df_metric = df_metric.dropna(subset=[selected_metric])
    df_metric_sorted = df_metric.sort_values(
        by=selected_metric,
        ascending=metric_sorting[selected_metric]["ascending"]
    )
    print(f"XGBoost models sorted by {selected_metric}:\n")
    print(df_metric_sorted)
    print('-------------------------------------')

# === Full registry top 5 ===
all_results = []

for model_key, model_info in trained_registry.items():
    metric_value = model_info.get("original_metrics", {}).get(selected_metric)
    all_results.append({
        "model_key": model_key,
        "model_name": model_info.get("model_name"),
        "dataset_name": model_info.get("dataset_name"),
        selected_metric: metric_value
    })

df_all = pd.DataFrame(all_results)
df_all = df_all.dropna(subset=[selected_metric])

top5_models = df_all.sort_values(
    by=selected_metric,
    ascending=metric_sorting[selected_metric]["ascending"]
).head(5)

print(f"\nTop 5 models by {selected_metric}:\n")
print(top5_models)

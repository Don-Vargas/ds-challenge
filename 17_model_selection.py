import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.storage import path_validate
from utils.registry import (
    load_registry,
    load_trained_registry
)

# === CONFIGURATION ===
# Choose metric: "rmse", "mae", or "r2"
selected_metric = "rmse"

# Define whether higher is better or not
# You can add more metrics as needed
metric_sorting = {
    "rmse": {"ascending": True},  # lower is better
    "mae": {"ascending": True},   # lower is better
    "r2": {"ascending": False}    # higher is better
}
# ======================

# Load registries
registry = load_registry()
trained_registry = load_trained_registry()

# Filter XGBoost models
xgboost_models = {
    k: v for k, v in trained_registry.items()
    if v.get("model_name") == "xgboost"
}

# Collect results
results = []

for model_key, model_info in xgboost_models.items():
    metric_value = model_info.get("original_metrics", {}).get(selected_metric, None)
    results.append({
        "model_key": model_key,
        "dataset_name": model_info.get("dataset_name"),
        selected_metric: metric_value
    })

# Create DataFrame
df_metric = pd.DataFrame(results)

# Sort by selected metric (using configured order)
df_metric_sorted = df_metric.sort_values(
    by=selected_metric,
    ascending=metric_sorting[selected_metric]["ascending"]
)

print(f"XGBoost models sorted by {selected_metric}:\n")
print(df_metric_sorted)
print('-------------------------------------')

# === Full registry results ===
results = []  # Reset

for model_key, model_info in trained_registry.items():
    metric_value = model_info.get("original_metrics", {}).get(selected_metric, None)
    results.append({
        "model_key": model_key,
        "model_name": model_info.get("model_name"),
        "dataset_name": model_info.get("dataset_name"),
        selected_metric: metric_value
    })

# Create full DataFrame
df_all = pd.DataFrame(results)

# Drop rows with missing values for the selected metric
df_all = df_all.dropna(subset=[selected_metric])

# Get top 5 models
top5_models = df_all.sort_values(
    by=selected_metric,
    ascending=metric_sorting[selected_metric]["ascending"]
).head(5)

print(f"\nTop 5 models by {selected_metric}:\n")
print(top5_models)

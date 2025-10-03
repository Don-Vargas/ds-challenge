import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from utils.registry import (
    load_registry,
    load_trained_registry,
    add_original_metrics_path_to_registry
)

# Load registries
registry = load_registry()
trained_registry = load_trained_registry()

def performance_metrics():
    for model_key, model_info in trained_registry.items():
        # Access the path to original scale test predictions
        pred_file = model_info.get("predictions_original_scale_test_csv_path")

        if not pred_file or not os.path.exists(pred_file):
            print(f"[SKIP] File not found for {model_key}: {pred_file}")
            continue

        # Load prediction CSV
        df = pd.read_csv(pred_file)

        # Ensure required columns exist
        if 'y_true' not in df.columns or 'y_pred' not in df.columns:
            print(f"[ERROR] Columns 'y_true' or 'y_pred' missing in {pred_file}")
            continue

        y_true = df['y_true']
        y_pred = df['y_pred']

        # Compute metrics
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        model_name = model_info["model_name"]
        dataset_name = model_info["dataset_name"]
        base_dataset_name = f"{model_name}_{dataset_name}"

        print(f"[METRICS] {base_dataset_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Save metrics to registry
        add_original_metrics_path_to_registry(base_dataset_name, [rmse, mae, r2])

if __name__ == '__main__':
    performance_metrics()
import os
import pandas as pd
import numpy as np

from utils.storage import load_pickle
from utils.registry import (
    load_registry, 
    load_trained_registry, 
    add_predictions_test_original_scale_data_path_to_registry,
    add_predictions_blind_original_scale_data_path_to_registry
)
from config.paths import (
    TEST_DATA_TRANSFORMED_PREDS_DIR,
    BLIND_DATA_TRANSFORMED_PREDS_DIR
)

# Load registries
registry = load_registry()
trained_registry = load_trained_registry()

# Split model name into base + suffix
def split_suffix(s, suffix_parts=2):
    parts = s.split('_')
    if len(parts) < suffix_parts:
        return s, ''
    prefix = '_'.join(parts[:-suffix_parts])
    suffix = '_'.join(parts[-suffix_parts:])
    return prefix, suffix

# Read predictions CSV
def read_csv_file(path):
    df = pd.read_csv(path)
    if 'y_true' not in df.columns:
        return df, None
    y_pred = df.drop(columns='y_true')
    y_true = df['y_true']
    return y_pred, y_true

# Reverses the transformation applied to target values
def reverse_transform(y, scaler_trans_model):
    if scaler_trans_model is None:
        return np.asarray(y).flatten()  # Always convert & flatten
    
    transformer = load_pickle(scaler_trans_model)

    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_array = y.to_numpy().reshape(-1, 1)
    else:
        y_array = np.asarray(y).reshape(-1, 1)

    reversed_y = transformer.inverse_transform(y_array).flatten()
    return reversed_y  # Always numpy 1D array


# Main function to reverse transformed predictions and save results
def reversed_transformed_preds(is_test=True, output_dir=None):
    for prefix, info in trained_registry.items():
        model_name = info.get("model_name")
        model, data_type = split_suffix(model_name)

        if is_test:
            preds_csv = info.get('predictions_test_csv_path')
            y_pred, y_true = read_csv_file(preds_csv)
        else:
            preds_csv = info.get('predictions_bind_csv_path')
            y_pred, y_true = read_csv_file(preds_csv)

        # Load transformer for y (if any)
        scaler_info = info.get("transformer_info", {}).get("scaler_pickle_path", None)
        if isinstance(scaler_info, dict):
            scaler_trans_model = scaler_info.get("y", None)
        elif isinstance(scaler_info, str) and scaler_info.lower() != "none":
            scaler_trans_model = scaler_info
        else:
            scaler_trans_model = None

        # Reverse prediction transformation
        reversed_y_pred = reverse_transform(y_pred, scaler_trans_model)

        # Compose output filename
        output_path = os.path.join(output_dir, f'preds_{prefix}.csv')

        if is_test:
            reversed_y_true = reverse_transform(y_true, scaler_trans_model) if y_true is not None else None
            df_preds = pd.DataFrame({
                "y_true": reversed_y_true if reversed_y_true is not None else None,
                "y_pred": reversed_y_pred
            })
            add_predictions_test_original_scale_data_path_to_registry(
                model_key=prefix, 
                predictions_csv_path=output_path
            )
        else:
            df_preds = pd.DataFrame({"y_pred": reversed_y_pred.flatten()})
            add_predictions_blind_original_scale_data_path_to_registry(
                model_key=prefix, 
                predictions_csv_path=output_path
            )

        df_preds.to_csv(output_path, index=False)
        print('###################################')

# Execution block
if __name__ == '__main__':
    
    reversed_transformed_preds(
        is_test=True,
        output_dir=TEST_DATA_TRANSFORMED_PREDS_DIR
    )
    
    reversed_transformed_preds(
        is_test=False,
        output_dir=BLIND_DATA_TRANSFORMED_PREDS_DIR
    )

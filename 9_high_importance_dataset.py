import os
import pandas as pd

from utils.registry import load_registry, add_most_important_data_path_to_registry
from config.paths import TRAINING_DATA_IMPORTANT_DIR, TEST_DATA_IMPORTANT_DIR, BLIND_DATA_IMPORTANT_DIR

IMPORTANT_FEATURES = ['feature_2', 'feature_9', 'feature_11', 'feature_13', 'feature_18']
TARGET_COLUMN = 'target'

def process_dataset_with_target():
    """
    Procesa datasets con target (train/test), filtra columnas importantes y guarda nuevos archivos.
    """
    registry = load_registry()
    output_folder_list=[TRAINING_DATA_IMPORTANT_DIR, TEST_DATA_IMPORTANT_DIR]

    for prefix, info in registry.items():
        if info.get("type") == 'blind':
            continue

        if info.get("type") == 'train':
            output_folder = output_folder_list[0]
        else:
            output_folder = output_folder_list[1]

        input_path = info.get("transformed_data_csv_path")
        if not input_path or not os.path.isfile(input_path):
            print(f"[WARNING] Skipping '{prefix}' — missing or invalid CSV path.")
            continue

        try:
            df = pd.read_csv(input_path)
            columns_to_keep = [col for col in IMPORTANT_FEATURES + [TARGET_COLUMN] if col in df.columns]
            df_filtered = df[columns_to_keep]

            output_filename = f"important_{os.path.basename(input_path)}"
            output_path = os.path.join(output_folder, output_filename)
            df_filtered.to_csv(output_path, index=False)

            add_most_important_data_path_to_registry(prefix, output_path)
            print(f"[INFO] Saved (with target): {output_filename}")

        except Exception as e:
            print(f"[ERROR] Failed processing '{prefix}': {e}")


def process_dataset_without_target():
    """
    Procesa datasets sin target (blind), filtra columnas importantes y guarda nuevos archivos.
    """
    registry = load_registry()
    output_folder=BLIND_DATA_IMPORTANT_DIR

    for prefix, info in registry.items():
        if not info.get("type") == 'blind':
            continue

        input_path = info.get("transformed_data_csv_path")
        if not input_path or not os.path.isfile(input_path):
            print(f"[WARNING] Skipping '{prefix}' — missing or invalid CSV path.")
            continue

        try:
            df = pd.read_csv(input_path)
            columns_to_keep = [col for col in IMPORTANT_FEATURES if col in df.columns]
            df_filtered = df[columns_to_keep]

            output_filename = f"important_{os.path.basename(input_path)}"
            output_path = os.path.join(output_folder, output_filename)
            df_filtered.to_csv(output_path, index=False)

            add_most_important_data_path_to_registry(prefix, output_path)
            print(f"[INFO] Saved (blind): {output_filename}")

        except Exception as e:
            print(f"[ERROR] Failed processing '{prefix}': {e}")


if __name__ == "__main__":
    process_dataset_with_target()
    process_dataset_without_target()

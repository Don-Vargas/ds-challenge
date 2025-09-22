import os
import pandas as pd

from utils.storage import path_validate
from utils.registry import load_registry, add_most_important_data_path_to_registry


def filter_and_save_important_features(output_folder, columns_to_keep):
    """
    Filters specified columns from registry-defined datasets and saves the filtered
    files to an output folder with a filename prefix.

    Args:
        registry (dict): Model registry mapping prefixes to info dicts.
        output_folder (str): Path to save filtered CSV files.
        columns_to_keep (list): List of column names to retain.
    """
    # Load model registry
    registry = load_registry()

    for prefix, info in registry.items():
        input_path = info.get("output_csv_path")

        if not input_path or not os.path.isfile(input_path):
            print(f"[WARNING] Skipping '{prefix}' — missing or invalid output_csv_path.")
            continue

        input_filename = os.path.basename(input_path)
        output_filename = f"important_{input_filename}"
        output_path = os.path.join(output_folder, output_filename)

        try:
            df = pd.read_csv(input_path, usecols=columns_to_keep)
            df.to_csv(output_path, index=False)

            # Correctly update the registry using the model key (prefix)
            add_most_important_data_path_to_registry(prefix, output_path)

            print(f"[INFO] Processed '{prefix}': saved -> {output_filename}")

        except ValueError as e:
            print(f"[WARNING] Skipped '{prefix}' — missing columns. Error: {e}")
        except Exception as e:
            print(f"[ERROR] Skipped '{prefix}' — unexpected error: {e}")


if __name__ == "__main__":

    # Define output folder and ensure it's ready
    output_folder = 'data/data_for_models/most_important_features/'
    path_validate(output_folder)

    # Define the most important columns
    columns_to_keep = ['feature_2', 'feature_9', 'feature_11', 'feature_13', 'feature_18', 'target']

    # Run the filtering
    filter_and_save_important_features(output_folder, columns_to_keep)

    print("[INFO] All filtering operations completed.")

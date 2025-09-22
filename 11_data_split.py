import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.storage import path_validate
from utils.registry import load_registry, add_split_data_path_to_registry


def split_and_save_datasets(datasets, output_dir, test_size=0.2, random_state=42):
    training_dir = os.path.join(output_dir, 'training')
    holdout_dir = os.path.join(output_dir, 'holdout')

    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    for name, path in datasets.items():
        if not os.path.isfile(path):
            print(f"[WARNING] File not found for '{name}': {path}")
            continue

        df = pd.read_csv(path)

        # Split into train/test (80/20)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        # Save to respective folders
        train_path = os.path.join(training_dir, f'{name}.csv')
        test_path = os.path.join(holdout_dir, f'{name}.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Update registry
        add_split_data_path_to_registry(name, [train_path, test_path])
        print(f"[INFO] Split saved for: {name}")


if __name__ == "__main__":
    output_dir = 'data/data_for_models/split_datasets/'
    path_validate(output_dir)

    # Load model registry
    registry = load_registry()
    datasets = {}

    for prefix, info in registry.items():
        paths_to_check = {
            "output_csv_path": info.get("output_csv_path"),
            "most_important_csv_path": info.get("most_important_csv_path"),
            "pca_data_csv_path": info.get("pca", {}).get("pca_data_csv_path")
        }

        for name, input_path in paths_to_check.items():
            if not input_path or not os.path.isfile(input_path):
                print(f"[WARNING] Skipping '{prefix}' — missing or invalid {name}.")
                continue

            datasets[prefix] = input_path
            print(datasets)

            split_and_save_datasets(datasets, output_dir)


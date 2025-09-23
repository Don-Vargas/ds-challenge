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
        add_split_data_path_to_registry(name.rsplit('_', 1)[0], [train_path, test_path])
        print(f"[INFO] Split saved for: {name}")

def data_splitter(output_dir):
    path_validate(output_dir)

    registry = load_registry()
    datasets = {}

    for prefix, info in registry.items():
        paths_to_check = {
            "original": info.get("output_csv_path"),
            "important": info.get("most_important_csv_path", {}),
            "pca": info.get("pca", {}).get("pca_data_csv_path")
        }

        for suffix, input_path in paths_to_check.items():
            if input_path and os.path.isfile(input_path):
                datasets[f"{prefix}_{suffix}"] = input_path
            else:
                print(f"[INFO] {prefix}_{suffix} no existe o es inválido, se ignora.")

    split_and_save_datasets(datasets, output_dir)

            
if __name__ == "__main__":
    output_dir = 'data/data_for_models/split_datasets/'
    data_splitter(output_dir)

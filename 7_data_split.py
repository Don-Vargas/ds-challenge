import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.storage import path_validate

def obtener_csvs(root_dir, excluded_file):
    csv_files = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
    dataset_dirs = {}

    excluded_file = os.path.normpath(excluded_file)
    excluded_name = os.path.basename(excluded_file)

    for file_path in csv_files:
        file_path_norm = os.path.normpath(file_path)
        file_name = os.path.basename(file_path_norm)

        if file_path_norm == excluded_file or file_name == excluded_name:
            continue

        key = os.path.splitext(file_name)[0]
        dataset_dirs[key] = file_path_norm

    return dataset_dirs

def split_and_save_datasets(datasets, output_dir, test_size=0.2, random_state=42):
    training_dir = os.path.join(output_dir, 'training')
    holdout_dir = os.path.join(output_dir, 'holdout')

    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    for name, path in datasets.items():
        df = pd.read_csv(path)

        # Split into train/test (80/20)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        # Save to respective folders
        train_path = os.path.join(training_dir, f'{name}.csv')
        test_path = os.path.join(holdout_dir, f'{name}.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

# Usage
root_dir = 'data/data_for_models/'
excluded_file = os.path.join(root_dir, 'blind_test_data.csv')
datasets = obtener_csvs(root_dir, excluded_file)

output_dir = 'data/data_for_models/split_datasets/'
path_validate(output_dir)
split_and_save_datasets(datasets, output_dir)

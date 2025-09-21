import os
import glob
import pandas as pd

from utils.storage import path_validate, TRAINING_DATA


def filter_and_save_important_features(file_paths, output_folder, columns_to_keep):
    """
    Filters specified columns from a list of CSV file paths and saves the filtered
    files to an output folder with a filename prefix.

    Args:
        file_paths (list): List of full paths to input CSV files.
        output_folder (str): Path to save filtered CSV files.
        columns_to_keep (list): List of column names to retain.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing {len(file_paths)} files...")

    for full_path in file_paths:
        file_name = os.path.basename(full_path)
        output_file = f"important_{file_name}"
        output_path = os.path.join(output_folder, output_file)

        try:
            # Read only selected columns
            df = pd.read_csv(full_path, usecols=columns_to_keep)

            # Save filtered DataFrame
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_file}")

        except ValueError as e:
            print(f"Skipped: {file_name} — missing required columns. Error: {e}")
        except Exception as e:
            print(f"Skipped: {file_name} — unexpected error: {e}")

if __name__ == "__main__":

    # Input: get full file paths
    input_folder = 'data/data_for_models/transformed_data/'
    file_paths = glob.glob(os.path.join(input_folder, '*.csv'))

    # Output folder
    output_folder = 'data/data_for_models/most_important_features/'
    path_validate(output_folder)

    # Columns to keep
    columns_to_keep = ['feature_2', 'feature_9', 'feature_11', 'feature_13', 'feature_18', 'target']

    # Call the function
    filter_and_save_important_features(file_paths, output_folder, columns_to_keep)

    file_paths = glob.glob(TRAINING_DATA)
    filter_and_save_important_features(file_paths, output_folder, columns_to_keep)

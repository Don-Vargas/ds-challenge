import os
import pandas as pd

from utils.storage import load_pickle
from utils.registry import (load_registry, 
                            load_trained_registry, 
                            add_predictions_test_data_path_to_registry,
                            add_predictions_blind_data_path_to_registry)
from config.paths import (TEST_DATA_RAW_PREDS_DIR,
                          BLIND_DATA_RAW_PREDS_DIR)

registry = load_registry()
trained_registry = load_trained_registry()

def split_suffix(s, suffix_parts=2):
    parts = s.split('_')
    if len(parts) < suffix_parts:
        # No hay suficientes partes para separar
        return s, ''
    
    prefix = '_'.join(parts[:-suffix_parts])
    suffix = '_'.join(parts[-suffix_parts:])
    return prefix, suffix

def read_csv_file(df):
    df = pd.read_csv(df)
    if 'target' not in df.columns:
        X = df
        return X
    else:
        X = df.drop(columns='target')
        y_true = df['target']
        return X, y_true


def generate_raw_preds(is_test=True, output_dir=None):
    for prefix, info in trained_registry.items():
        model_name = info.get("model_name")
        model, data_type = split_suffix(model_name)
        data_set_info = registry[info.get("dataset_name")]

        # Determine which dataset to load
        if data_type == 'most_important':
            df_path = data_set_info['most_important_csv_path']
        elif data_type == 'pca_data':
            df_path = data_set_info['pca']['pca_data_csv_path']
        elif data_type == 'transformed_data':
            df_path = data_set_info['transformed_data_csv_path']
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        # Construct the new path (switching training to test/blind)
        base_path, filename = os.path.split(df_path)
        # Replace 'B_training_data' with appropriate dataset type
        if is_test:
            base_path = base_path.replace('B_training_data', 'C_test_data')
            new_filename = filename.replace('train', 'test')
        else:
            base_path = base_path.replace('B_training_data', 'D_blind_data')
            new_filename = filename.replace('train', 'blind')

        new_df_path = os.path.join(base_path, new_filename)

        # Read data depending on test/blind mode
        if is_test:
            X, y_true = read_csv_file(new_df_path)
        else:
            print(new_df_path)
            X = read_csv_file(new_df_path)

        # Load model and predict
        prediction_model_path = info.get("model_pickle_path")
        prediction_model = load_pickle(prediction_model_path)
        y_pred = prediction_model.predict(X)

        preds_csv = f'{output_dir}preds_{prefix}.csv'
        # Create predictions DataFrame
        if is_test:
            df_preds = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred
            })
            add_predictions_test_data_path_to_registry(model_key=prefix, 
                                                       predictions_csv_path=preds_csv)
        else:
            df_preds = pd.DataFrame({"y_pred": y_pred})
            add_predictions_blind_data_path_to_registry(model_key=prefix, 
                                                        predictions_csv_path=preds_csv)

        # Write to appropriate output directory
        df_preds.to_csv(preds_csv, index=False)

if __name__ == '__main__':
    generate_raw_preds(
    is_test=True,
    output_dir=TEST_DATA_RAW_PREDS_DIR
    )
    
    generate_raw_preds(
        is_test=False,
        output_dir=BLIND_DATA_RAW_PREDS_DIR
    )

import os
import pickle
import pandas as pd
import numpy as np

from utils.storage import path_validate, load_pickle
from utils.registry import load_registry, load_trained_registry, add_predictions_data_path_to_registry

def predictions_transformed_scale(blind=False):
    trained_registry = load_trained_registry()

    if blind:
        y_pred_path = 'data/data_for_models/blind_data_preds/'
        path_validate(y_pred_path)
    else:
        # Create output directory
        repository_path = 'data/data_for_models/split_datasets/holdout/'
        predictions_csv_dir = 'data/predictions/transformed_scale/'
        path_validate(predictions_csv_dir)

    for model_key, model_info in trained_registry.items():
        model_path = model_info["model_pickle_path"]
        dataset_name = model_info["dataset_name"]
        model_name = model_info["model_name"]

        hold_out_path = f'{repository_path}{dataset_name}.csv'

        # Load holdout data
        df = pd.read_csv(hold_out_path)
        if 'target' not in df.columns:
            print(f"'target' column not found in {hold_out_path}")
            model = load_pickle(model_path)
            y_pred = model.predict(df)
            df_preds = pd.DataFrame({"y_pred": y_pred})
            df_preds.to_csv(y_pred_path, index=False)
            continue

        X_holdout = df.drop(columns='target')
        y_holdout = df['target']

        # Load model
        try:
            model = load_pickle(model_path)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            continue

        try:
            y_pred = model.predict(X_holdout)

            # Generate prediction DataFrame
            df_preds = pd.DataFrame({
                "y_true": y_holdout,
                "y_pred": y_pred
            })

            # Generate filename
            predictions_csv_filename = f"predictions_{dataset_name}_{model_name}.csv"
            predictions_csv_path = os.path.join(predictions_csv_dir, predictions_csv_filename)

            # Save predictions
            df_preds.to_csv(predictions_csv_path, index=False)

            # Register prediction path
            add_predictions_data_path_to_registry(f'{model_name}_{dataset_name}', predictions_csv_path)
        except Exception as e:
            print(f"Error predicting with model {model_name} on dataset {dataset_name}: {e}")
            continue

        print(f"Predictions done for model: {model_name} | dataset: {dataset_name}")



# Run
predictions_transformed_scale()
#predictions_transformed_scale(blind=True)

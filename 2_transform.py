import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

from utils.registry import add_transformation_record
from utils.storage import save_pickle
from config.paths import (
    TRAINING_DATA_TRANSFORMED_DIR,
    TRAIN_SPLIT_RAW_FILE,
    TEST_DATA_TRANSFORMED_DIR,
    TEST_SPLIT_RAW_FILE,
    BLIND_DATA_TRANSFORMED_DIR,
    BLIND_RAW_FILE,
    TRAINING_MODEL_TRANSFORMED_DIR,
    TEST_MODEL_TRANSFORMED_DIR,
    BLIND_MODEL_TRANSFORMED_DIR
)
# ----------------------------
# PATH FILE CONFIG
# ----------------------------
def get_data_config():
    return {
        'train': {
            'transformed_data_path': TRAINING_DATA_TRANSFORMED_DIR,
            'transformed_model_path': TRAINING_MODEL_TRANSFORMED_DIR,
            'file': TRAIN_SPLIT_RAW_FILE,
            'blind': False
        },
        'test': {
            'transformed_data_path': TEST_DATA_TRANSFORMED_DIR,
            'transformed_model_path': TEST_MODEL_TRANSFORMED_DIR,
            'file': TEST_SPLIT_RAW_FILE,
            'blind': False
        },
        'blind': {
            'transformed_data_path': BLIND_DATA_TRANSFORMED_DIR,
            'transformed_model_path': BLIND_MODEL_TRANSFORMED_DIR,
            'file': BLIND_RAW_FILE,
            'blind': True
        }
    }
# ----------------------------
# TRANSFORMERS
# ----------------------------
def get_available_transformers():
    return {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'boxcox': PowerTransformer(method='box-cox'),
        'yeojohnson': PowerTransformer(method='yeo-johnson')
    }

def select_transformers_by_name(transformer_names=None):
    all_transformers = get_available_transformers()
    if transformer_names is None:
        return all_transformers
    else:
        return {
            name: all_transformers[name]
            for name in transformer_names
            if name in all_transformers
        }

# ----------------------------
# TRANSFORM FUNCTIONS
# ----------------------------
import pandas as pd
import os

def apply_transform_and_save(df, transformer, output_path, feature_names=None):
    transformed = transformer.fit_transform(df)
    df_transformed = pd.DataFrame(transformed, columns=feature_names if feature_names is not None else df.columns)
    df_transformed.to_csv(output_path, index=False)
    return df_transformed

def complete_data_transform(df, transformer_X, transformer_y, file_full_path, transformed_model_path, output_prefix):
    X = df.drop(columns=['target'])
    y = df[['target']]

    # Apply transformations and save data
    df_transformed_X = apply_transform_and_save(X, transformer_X, file_full_path, X.columns)
    y_transformed = transformer_y.fit_transform(y)
    df_transformed_X['target'] = y_transformed
    df_transformed_X.to_csv(file_full_path, index=False)

    # Save transformers
    scaler_X_path = os.path.join(transformed_model_path, f"scaler_X_{output_prefix}.pkl")
    scaler_y_path = os.path.join(transformed_model_path, f"scaler_y_{output_prefix}.pkl")
    save_pickle(transformer_X, scaler_X_path)
    save_pickle(transformer_y, scaler_y_path)

    return {"X": scaler_X_path, "y": scaler_y_path}

def blind_data_transform(df, transformer_X, file_full_path, transformed_model_path, output_prefix):
    apply_transform_and_save(df, transformer_X, file_full_path, df.columns)

    scaler_X_path = os.path.join(transformed_model_path, f"scaler_X_{output_prefix}.pkl")
    save_pickle(transformer_X, scaler_X_path)

    return {"X": scaler_X_path}

def split_datasets_register():
    data_config = get_data_config()

    # Iterate over each data type (train, test, blind)
    for key, config in data_config.items():
        blind = config['blind']
        transformed_data_path = config['transformed_data_path']
        file_full_path = f'{transformed_data_path}{key}.csv'
        # Registro
        add_transformation_record(
            output_prefix=key,
            transformer='',
            transformed_data_csv_path=file_full_path,
            scaler_pickle_path='None',
            blind=blind
        )

def transform(transformer_names = None):
    split_datasets_register()
    data_config = get_data_config()

    # Iterate over each data type (train, test, blind)
    for key, config in data_config.items():
        blind = config['blind']
        file = config['file']
        transformed_data_path = config['transformed_data_path']
        transformed_model_path = config['transformed_model_path']

        # Load data
        df = pd.read_csv(file, index_col=0)

        # Choose transformers
        if blind:
            transformer_names
        else:
            transformer_names = None
        
        selected_transformers = select_transformers_by_name(transformer_names)

        for name, transformer in selected_transformers.items():
            output_prefix = name
            file_full_path = f'{transformed_data_path}{output_prefix}.csv'

            if not blind:
                # Explicitly pass separate transformer for y
                scaler_pickle_path = complete_data_transform(
                    df,
                    transformer_X=transformer,
                    transformer_y=transformer.__class__(),  # Separate instance for target
                    file_full_path=file_full_path,
                    transformed_model_path=transformed_model_path,
                    output_prefix=output_prefix
                )
            else:
                scaler_pickle_path = blind_data_transform(
                    df,
                    transformer_X=transformer,
                    file_full_path=file_full_path,
                    transformed_model_path=transformed_model_path,
                    output_prefix=f'{key}_{output_prefix}'
                )

            # Registro
            add_transformation_record(
                output_prefix=f'{key}_{output_prefix}',
                transformer=transformer,
                transformed_data_csv_path=file_full_path,
                scaler_pickle_path=scaler_pickle_path,
                blind=blind
            )

# ----------------------------
# MAIN EXECUTION BLOCK
# ----------------------------
if __name__ == "__main__":
    transform()

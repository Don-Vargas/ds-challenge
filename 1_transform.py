import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

from utils.registry import add_transformation_record
from utils.storage import save_pickle, path_validate, TRAINING_DATA

def transform_and_save(df, transformer, output_prefix):
    transformed_data_path = 'data/data_for_models/transformed_data/'
    scaler_path = 'data/scaler_models/'
    
    path_validate(transformed_data_path)
    path_validate(scaler_path)
    
    file_full_path = f'{transformed_data_path}{output_prefix}.csv'
    
    if transformer:
        scaler_X = transformer
        scaler_y = transformer.__class__()  # Nuevo objeto del mismo tipo
        
        # Separar X e y
        X = df.drop(columns=['target'])
        y = df[['target']]
        
        # Transformar
        X_transformed = scaler_X.fit_transform(X)
        y_transformed = scaler_y.fit_transform(y)

        # Reconstruir DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=X.columns)
        df_transformed['target'] = y_transformed

        # Guardar CSV
        df_transformed.to_csv(file_full_path, index=False)

        # Guardar scalers
        save_pickle(scaler_X, f"{scaler_path}scaler_X_{output_prefix}.pkl")
        save_pickle(scaler_y, f"{scaler_path}scaler_y_{output_prefix}.pkl")

        scaler_pickle_path = {
            "X": f"{scaler_path}scaler_X_{output_prefix}.pkl",
            "y": f"{scaler_path}scaler_y_{output_prefix}.pkl"
        }
    else:
        df.to_csv(file_full_path, index=False)
        scaler_pickle_path = None

    # Registro
    add_transformation_record(
        output_prefix=output_prefix,
        transformer=transformer,
        output_csv_path=file_full_path,
        scaler_pickle_path=scaler_pickle_path
    )


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)

    # Guardar el dataset original
    transform_and_save(df, transformer=None, output_prefix='training_data_original')

    # Diccionario de transformadores
    transformers = {
        'training_data_minmax': MinMaxScaler(),
        'training_data_standard': StandardScaler(),
        'training_data_boxcox': PowerTransformer(method='box-cox'),
        'training_data_yeojohnson': PowerTransformer(method='yeo-johnson'),
    }

    # Aplicar cada transformación
    for prefix, transformer in transformers.items():
        print(f"Aplicando transformación: {prefix}")
        transform_and_save(df, transformer, prefix)

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
        df_transformed = transformer.fit_transform(df)
        df_transformed = pd.DataFrame(df_transformed, columns=df.columns)
        df_transformed.to_csv(file_full_path, index=False)
        save_pickle(transformer, f"{scaler_path}scaler_{output_prefix}.pkl")
    else:
        df.to_csv(file_full_path, index=False)
    
    add_transformation_record(
    output_prefix=output_prefix,
    transformer=transformer,
    columns=list(df.columns),
    output_csv_path = file_full_path,
    scaler_pickle_path=f"{scaler_path}scaler_{output_prefix}.pkl" if transformer else None
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

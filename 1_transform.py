import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

from utils.registry import add_transformation_record
from utils.storage import save_pickle, path_validate, TRAINING_DATA, BLIND_DATA

def transform_and_save(df, transformer, output_prefix, transformed_data_path, blind=False):
    scaler_path = 'data/scaler_models/'
    
    path_validate(transformed_data_path)
    path_validate(scaler_path)
    
    file_full_path = f'{transformed_data_path}{output_prefix}.csv'
    
    if transformer:
        scaler_X = transformer
        if not blind:
            scaler_y = transformer.__class__()  # Nuevo objeto del mismo tipo
        
        # Separar X e y
        X = df
        if not blind:
            X = df.drop(columns=['target'])
            y = df[['target']]
        
        # Transformar
        X_transformed = scaler_X.fit_transform(X)
        if not blind:
            y_transformed = scaler_y.fit_transform(y)

        # Reconstruir DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=X.columns)
        if not blind:
            df_transformed['target'] = y_transformed

        # Guardar CSV
        df_transformed.to_csv(file_full_path, index=False)

        # Guardar scalers
        save_pickle(scaler_X, f"{scaler_path}scaler_X_{output_prefix}.pkl")
        if not blind:
            save_pickle(scaler_y, f"{scaler_path}scaler_y_{output_prefix}.pkl")


        if not blind:
            scaler_pickle_path = {
                "X": f"{scaler_path}scaler_X_{output_prefix}.pkl",
                "y": f"{scaler_path}scaler_y_{output_prefix}.pkl"
            }
        
        scaler_pickle_path = {
            "X": f"{scaler_path}scaler_X_{output_prefix}.pkl"
        } 
    else:
        df.to_csv(file_full_path, index=False)
        scaler_pickle_path = None

    # Registro
    add_transformation_record(
        output_prefix=output_prefix,
        transformer=transformer,
        output_csv_path=file_full_path,
        scaler_pickle_path=scaler_pickle_path,
        blind=blind
    )

def obtener_transformadores_disponibles():
    return {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'boxcox': PowerTransformer(method='box-cox'),
        'yeojohnson': PowerTransformer(method='yeo-johnson'),
    }

def apply_transformations(
    df: pd.DataFrame,
    transformed_data_path: str,
    output_prefix: str,
    blind: bool = False,
    include_original: bool = True,
    transformer_names: list = None  # <-- nuevo parámetro
):
    """
    Aplica transformaciones al DataFrame y guarda los resultados con diferentes prefijos.

    Parameters:
        df (pd.DataFrame): El DataFrame original.
        output_prefix (str): Prefijo base para los archivos de salida.
        blind (bool): Indica si los datos no tienen target (True) o sí (False).
        include_original (bool): Si se desea guardar el dataset original sin transformar.
        transformer_names (list): Lista opcional con los nombres de transformaciones a aplicar.
    """

    transformadores_disponibles = obtener_transformadores_disponibles()

    if transformer_names is None:
        # Aplica todas las transformaciones
        selected_transformers = transformadores_disponibles
    else:
        # Solo las seleccionadas (valida que existan)
        selected_transformers = {
            name: transformadores_disponibles[name]
            for name in transformer_names
            if name in transformadores_disponibles
        }

    # Guardar el dataset original si se solicita
    if include_original:
        transform_and_save(df, transformer=None, output_prefix=f'{output_prefix}_original', transformed_data_path=transformed_data_path, blind=blind)

    # Aplicar cada transformación
    for name, transformer in selected_transformers.items():
        full_prefix = f"{output_prefix}_{name}"
        print(f"Aplicando transformación: {full_prefix}")
        transform_and_save(df, transformer, output_prefix=full_prefix, transformed_data_path=transformed_data_path, blind=blind)

if __name__ == "__main__":
    transformed_data_path = 'data/data_for_models/transformed_data/'
    df = pd.read_csv(TRAINING_DATA)
    apply_transformations(df, transformed_data_path, output_prefix='training_data', blind=False)

    
    #transformed_data_path = 'data/data_for_models/blind_data_sets/'
    #df = pd.read_csv(BLIND_DATA)
    
    # Ejemplo 1: aplicar todas las transformaciones
    #transformed_data_path = 'data/data_for_models/transformed_data/'
    #apply_transformations(df, transformed_data_path, output_prefix='blind_data', blind=True)

    # Ejemplo 2: aplicar solo algunas
    #apply_transformations(df, transformed_data_path, output_prefix='blind_data', blind=True, transformer_names=['minmax', 'standard'])


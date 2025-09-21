import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, Normalizer, PowerTransformer

from utils.storage import save_pickle, path_validate, TRAINING_DATA


def explorar_dataset(df, incluir_categoricas=False):
    """
    Genera un resumen estadístico por columna del DataFrame.
    
    Parámetros:
    - df: pandas.DataFrame
    - incluir_categoricas: bool, si True, incluye columnas no numéricas con conteos básicos
    
    Retorna:
    - reporte: pandas.DataFrame
    """
    reporte = []

    for columna in df.columns:
        serie = df[columna]
        tipo = serie.dtype

        if np.issubdtype(tipo, np.number):
            skewness = stats.skew(serie.dropna())
            kurtosis = stats.kurtosis(serie.dropna())
            shapiro_test = stats.shapiro(serie.dropna())
            p_shapiro = shapiro_test.pvalue
            es_normal_shapiro = 1 if p_shapiro > 0.05 else 0
            media = serie.dropna().mean()
            std = serie.dropna().std()
            ks_stat, p_ks = stats.kstest(serie.dropna(), 'norm', args=(media, std))
            es_normal_ks = 1 if p_ks > 0.05 else 0

            estadisticas = {
                'Columna': columna,
                'Tipo': tipo,
                'Valores únicos': serie.nunique(),
                'Nulos': serie.isnull().sum(),
                'Min': serie.min(),
                'Max': serie.max(),
                'Media': serie.mean(),
                'Mediana': serie.median(),
                'Moda': serie.mode().iloc[0] if not serie.mode().empty else np.nan,
                'Varianza': serie.var(),
                'Desv. Estándar': serie.std(),
                'Sesgo': skewness,
                'Curtosis': kurtosis,
                'p-valor Shapiro': p_shapiro,
                'Es normal (Shapiro)': es_normal_shapiro,
                'p-valor KS': p_ks,
                'Es normal (KS)': es_normal_ks,
            }
        elif incluir_categoricas:
            estadisticas = {
                'Columna': columna,
                'Tipo': tipo,
                'Valores únicos': serie.nunique(),
                'Nulos': serie.isnull().sum(),
                'Moda': serie.mode().iloc[0] if not serie.mode().empty else np.nan,
                'Valor más frecuente (conteo)': serie.value_counts().iloc[0] if not serie.value_counts().empty else np.nan,
            }
        else:
            continue

        reporte.append(estadisticas)

    return pd.DataFrame(reporte)

def process_and_report(df, transformer, output_prefix, incluir_categoricas=True):
    """
    Aplica una transformación opcional a un DataFrame, guarda los datos transformados 
    y genera un reporte de análisis exploratorio (EDA).

    Parámetros:
    ----------
    df : pandas.DataFrame
        El DataFrame de entrada que será procesado.
    
    transformer : object o None
        Un transformador compatible con scikit-learn que implemente los métodos 
        `fit_transform()`, como MinMaxScaler o Normalizer. Si se pasa `None`, 
        no se aplica ninguna transformación al DataFrame original.
    
    output_prefix : str
        Prefijo usado para nombrar los archivos de salida. Se generarán dos archivos:
        - data/{output_prefix}.csv: el DataFrame transformado (o el original si no hay transformación).
        - data/eda_{output_prefix}.csv: el reporte EDA generado por la función `explorar_dataset`.
    
    incluir_categoricas : bool, opcional (default=True)
        Indica si se deben incluir variables categóricas en el análisis exploratorio.

    Retorna:
    -------
    None
        La función no retorna ningún valor, pero guarda archivos CSV con los resultados.
    """

    transformed_data_path = 'data/data_for_models/transformed_data/'
    path_validate(transformed_data_path)
    scaler_path = 'data/scaler_models/'
    path_validate(scaler_path)
    eda_path = 'data/eda/'
    path_validate(eda_path)

    if transformer:
        df_transformed = transformer.fit_transform(df)
        df_transformed = pd.DataFrame(df_transformed, columns=df.columns)
        df_transformed.to_csv(f'{transformed_data_path}{output_prefix}.csv', index=False)
        save_pickle(transformer, f"{scaler_path}scaler_{output_prefix}.pkl")
    else:
        df_transformed = df
    reporte = explorar_dataset(df_transformed, incluir_categoricas=incluir_categoricas)
    #print(reporte)
    reporte.round(10).to_csv(f'{eda_path}eda_{output_prefix}.csv', index=False)

df = pd.read_csv(TRAINING_DATA)
process_and_report(df, transformer=False, output_prefix='training_data')
# Process using different transformations
process_and_report(df, MinMaxScaler(), 'training_data_minmax')
process_and_report(df, Normalizer(norm='l1'), 'training_data_l1')
process_and_report(df, Normalizer(norm='l2'), 'training_data_l2')
process_and_report(df, PowerTransformer(method='box-cox'), 'training_data_boxcox')
process_and_report(df, PowerTransformer(method='yeo-johnson'), 'training_data_yeojohnson')

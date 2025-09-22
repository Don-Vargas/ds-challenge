import numpy as np
import pandas as pd
from scipy import stats

from utils.storage import path_validate
from utils.registry import load_registry, add_eda_path_to_registry

def explorar_dataset(df, incluir_categoricas=False):
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

def generate_report(input_csv, output_prefix, eda_path, incluir_categoricas=True):
    df = pd.read_csv(input_csv)
    reporte = explorar_dataset(df, incluir_categoricas=incluir_categoricas)
    reporte.round(10).to_csv(f'{eda_path}eda_{output_prefix}.csv', index=False)

if __name__ == "__main__":
    registry = load_registry()
    eda_path = 'data/eda/'
    path_validate(eda_path)

    for prefix, info in registry.items():
        if "output_csv_path" not in info:
            print(f"[WARNING] No se encontró 'output_csv_path' para '{prefix}'. Saltando...")
            continue

        file_path = info["output_csv_path"]
        print(f"Generando reporte para {prefix} desde {file_path}")
        
        # Generar el reporte
        generate_report(file_path, prefix, eda_path, incluir_categoricas=True)

        # Generar y registrar la ruta al archivo EDA
        eda_report_path = f"{eda_path}eda_{prefix}.csv"
        add_eda_path_to_registry(prefix, eda_report_path)

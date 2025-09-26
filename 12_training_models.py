import os
import pandas as pd

from utils.training_utils import definir_metricas, definir_modelos, procesar_dataset
from utils.registry import load_registry
from config.paths import GRID_SEARCH_MODELS_PATH

def dataset_fields():
    return [
        ("transformed_data_csv_path", None),
        ("most_important_csv_path", None),
        ("pca_data_csv_path", "pca")
    ]

def data_trainer():
    registry = load_registry()
    scoring = definir_metricas()
    models = definir_modelos()
    resultados_totales = []

    for prefix, info in registry.items():
        if info.get("type") != 'train':
            continue

        for field, subfield in dataset_fields():
            file_path = info.get(field) if not subfield else info.get(subfield, {}).get(field)

            if not file_path or not os.path.exists(file_path):
                print(f"Archivo no encontrado: {file_path} (campo: {field}) — Saltando.")
                continue

            print(f"Procesando archivo: {file_path}")

            print(f" Procesando dataset: {prefix}")
            
            resultados = procesar_dataset(prefix, file_path, models, scoring)
            resultados_totales.extend(resultados)

    results_df = pd.DataFrame(resultados_totales).sort_values(by='best_rmse')
    output_path = f"{GRID_SEARCH_MODELS_PATH}resultados_gridsearch.csv"
    # output_path = os.path.join(GRID_SEARCH_PATH, "resultados_gridsearch.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n Resultados guardados en: {output_path}")

if __name__ == "__main__":
    data_trainer()


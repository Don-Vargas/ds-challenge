import pandas as pd
from utils.storage import path_validate
from utils.training_utils import definir_metricas, obtener_csvs, definir_modelos, procesar_dataset

from config.paths import TRAIN_DIR

def data_trainer():
    path = f'{TRAIN_DIR}/datasets/'
    scoring = definir_metricas()
    datasets = obtener_csvs(path)
    models = definir_modelos()
    
    resultados_totales = []
    for nombre, ruta in datasets.items():
        print(f"\nProcesando dataset: {nombre}")
        
        resultados = procesar_dataset(nombre.removesuffix(".csv"), ruta, models, scoring)
        resultados_totales.extend(resultados)

    results_df = pd.DataFrame(resultados_totales).sort_values(by='best_rmse')

    print("\nRESULTADOS RESUMIDOS:")
    print(results_df[['dataset', 'model', 'best_rmse', 'best_mae', 'best_r2', 'best_params']])

    gridsearch_path = 'data/grid_search/'
    path_validate(gridsearch_path)
    results_df.to_csv(f"{gridsearch_path}resultados_gridsearch.csv", index=False)


if __name__ == "__main__":
    data_trainer()


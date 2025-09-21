import pandas as pd
from utils.vis_utils import visualizar_resultados

if __name__ == "__main__":
    gridsearch_path = 'data/grid_search/resultados_gridsearch.csv'

    results_df = pd.read_csv(gridsearch_path)
    print("\nResultados cargados desde CSV:")
    print(results_df.head())

    visualizar_resultados(results_df)
    print(f"\nVisualizaciones guardadas en data/grid_search/")

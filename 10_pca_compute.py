import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.storage import path_validate, save_pickle
from utils.registry import load_registry, add_pca_data_path_to_registry


def select_pca_components(df, output_prefix, variance_threshold=0.80):
    """
    Aplica PCA a un DataFrame y guarda los datos transformados y el modelo entrenado.
    """
    y = df['target']
    X = df.drop(columns=['target'])

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    pca = PCA()
    X_pca_full = pca.fit_transform(X_arr)

    # Guardar el modelo PCA
    pca_model_path = f'data/pca_models/pca_model_{output_prefix}.pkl'
    save_pickle(pca, pca_model_path)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_selected = np.argmax(cumulative >= variance_threshold) + 1

    X_pca_sel = X_pca_full[:, :n_selected]
    col_names = [f'PC_{i+1}' for i in range(n_selected)]
    df_pca = pd.DataFrame(X_pca_sel, columns=col_names)
    df_pca['target'] = y

    # Guardar CSV
    pca_data_path = 'data/data_for_models/pca_data/'
    path_validate(pca_data_path)
    output_csv_path = os.path.join(pca_data_path, f'pca_{output_prefix}.csv')
    df_pca.to_csv(output_csv_path, index=False)
    add_pca_data_path_to_registry(output_prefix,[output_csv_path,pca_model_path])

    return df_pca, pca, n_selected, cumulative, output_csv_path


if __name__ == "__main__":
    registry = load_registry()
    for prefix, info in registry.items():
        input_path = info.get("output_csv_path")

        if not input_path or not os.path.isfile(input_path):
            print(f"[WARNING] Skipping '{prefix}' — missing or invalid output_csv_path.")
            continue

        try:
            df = pd.read_csv(input_path)
            df_pca, pca_model, n_components, cumulative_var, output_csv_path = select_pca_components(
                df, output_prefix=prefix, variance_threshold=0.80
            )
        
        except Exception as e:
            print(f"[ERROR] PCA failed for '{prefix}': {str(e)}")

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.storage import save_pickle
from utils.registry import load_registry, add_pca_data_path_to_registry
from config.paths import (
    TRAINING_DATA_PCA_DIR, TEST_DATA_PCA_DIR, BLIND_DATA_PCA_DIR,
    TRAINING_MODEL_PCA_DIR, TEST_MODEL_PCA_DIR, BLIND_MODEL_PCA_DIR
)

VARIANCE_THRESHOLD = 0.80


def load_and_validate_csv(path):
    if not path or not os.path.isfile(path):
        print(f"[WARNING] Skipping — missing or invalid CSV path: {path}")
        return None
    return pd.read_csv(path)


def apply_pca(X, variance_threshold=VARIANCE_THRESHOLD):
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_selected = np.argmax(cumulative >= variance_threshold) + 1
    X_pca_sel = X_pca_full[:, :n_selected]
    return pca, X_pca_sel, n_selected


def save_pca_outputs(prefix, df_pca, pca_model, output_data_folder, output_model_folder):
    pca_model_path = os.path.join(output_model_folder, f'pca_model_{prefix}.pkl')
    output_csv_path = os.path.join(output_data_folder, f'pca_{prefix}.csv')

    save_pickle(pca_model, pca_model_path)
    df_pca.to_csv(output_csv_path, index=False)
    add_pca_data_path_to_registry(prefix, [output_csv_path, pca_model_path])


def process_and_save(prefix, info, with_target, output_data_folder, output_model_folder):
    input_path = info.get("transformed_data_csv_path")
    df = load_and_validate_csv(input_path)
    if df is None:
        return

    try:
        if with_target:
            y = df['target']
            X = df.drop(columns=['target'])
        else:
            X = df

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        pca, X_pca_sel, n_selected = apply_pca(X_arr)
        col_names = [f'PC_{i+1}' for i in range(n_selected)]
        df_pca = pd.DataFrame(X_pca_sel, columns=col_names)

        if with_target:
            df_pca['target'] = y

        save_pca_outputs(prefix, df_pca, pca, output_data_folder, output_model_folder)

    except Exception as e:
        print(f"[ERROR] Failed processing '{prefix}': {e}")


def pca_with_target():
    registry = load_registry()
    for prefix, info in registry.items():
        if info.get("type") == 'blind':
            continue

        if info.get("type") == 'train':
            data_dir = TRAINING_DATA_PCA_DIR
            model_dir = TRAINING_MODEL_PCA_DIR
        else:
            data_dir = TEST_DATA_PCA_DIR
            model_dir = TEST_MODEL_PCA_DIR

        process_and_save(prefix, info, with_target=True,
                         output_data_folder=data_dir,
                         output_model_folder=model_dir)


def pca_with_no_target():
    registry = load_registry()
    for prefix, info in registry.items():
        if info.get("type") != 'blind':
            continue

        process_and_save(prefix, info, with_target=False,
                         output_data_folder=BLIND_DATA_PCA_DIR,
                         output_model_folder=BLIND_MODEL_PCA_DIR)


if __name__ == "__main__":
    pca_with_target()
    pca_with_no_target()
    print("[INFO] Proceso PCA completado.")

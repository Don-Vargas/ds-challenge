import os
import pandas as pd

from utils.storage import path_validate
from utils.registry import load_registry, add_most_important_data_path_to_registry


def filter_and_save_important_features(output_folder, features, include_target, target_column='target', blind_filter=None):
    """
    Filtra columnas importantes de datasets en el registry, con filtro opcional por 'blind'.
    """
    registry = load_registry()

    for prefix, info in registry.items():

        # Filtrar según blind_filter
        is_blind = info.get("blind", False)  # False si no está definido
        if blind_filter is not None and is_blind != blind_filter:
            continue  # Saltar datasets que no coincidan con filtro blind

        input_path = info.get("output_csv_path")
        if not input_path or not os.path.isfile(input_path):
            print(f"[WARNING] Skipping '{prefix}' — missing or invalid output_csv_path.")
            continue

        # Definir columnas a mantener para este dataset
        columns_to_keep = features.copy()
        if include_target and not is_blind and target_column not in columns_to_keep:
            columns_to_keep.append(target_column)

        # Leer solo las columnas existentes (evita errores si target no existe)
        try:
            df_all = pd.read_csv(input_path)
            existing_columns = [col for col in columns_to_keep if col in df_all.columns]
            df_filtered = df_all[existing_columns]

            # Guardar CSV filtrado
            output_filename = f"important_{os.path.basename(input_path)}"
            output_path = os.path.join(output_folder, output_filename)
            df_filtered.to_csv(output_path, index=False)

            add_most_important_data_path_to_registry(prefix, output_path)
            print(f"[INFO] Processed '{prefix}': saved -> {output_filename}")

        except Exception as e:
            print(f"[ERROR] Skipped '{prefix}' — unexpected error: {e}")


def run_feature_filtering(output_folder: str,
                          features: list,
                          include_target: bool = True,
                          target_column: str = 'target',
                          blind_filter: bool = None):
    """
    Ejecuta el filtrado con opción de incluir target y filtrar datasets por blind/no blind.
    """
    path_validate(output_folder)
    filter_and_save_important_features(output_folder, features, include_target, target_column, blind_filter)
    print("[INFO] Feature filtering process completed.")


if __name__ == "__main__":

    important_features = ['feature_2', 'feature_9', 'feature_11', 'feature_13', 'feature_18']

    # Procesar solo blind datasets (sin target)
    run_feature_filtering(
        output_folder='data/data_for_models/blind_data_sets/',
        features=important_features,
        include_target=False,
        blind_filter=True
    )

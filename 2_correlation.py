import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.storage import path_validate, TRAINING_DATA


def correlation_matrix_pearson(df, output_prefix, save_path): 
    """
    Calcula la matriz de correlación de Pearson, guarda CSV y heatmap.
    """
    correlation_matrix = df.corr(method='pearson')
    sorted_index = correlation_matrix["target"].sort_values(ascending=False).index
    sorted_corr_matrix = correlation_matrix.loc[sorted_index, sorted_index]

    # Guardar CSV
    csv_file = f'{save_path}{output_prefix}.csv'
    sorted_corr_matrix.round(2).to_csv(csv_file, index=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sorted_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Pearson Matrix correlation {output_prefix}")
    plt.savefig(f"{save_path}correlation_matrix_pearson_{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return sorted_corr_matrix.index

correlation_matrices_path = 'data/correlation_matrices/'
path_validate(correlation_matrices_path)

folder_path = 'data/data_for_models/transformed_data/'
file_names = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Calcular matrices de correlación para todos los CSV
target_corr = [
    correlation_matrix_pearson(
        pd.read_csv(os.path.join(folder_path, fname)),
        output_prefix=os.path.splitext(fname)[0],
        save_path=correlation_matrices_path
    )
    for fname in file_names
]

df = pd.read_csv(TRAINING_DATA)
df_corr = correlation_matrix_pearson(df, output_prefix='training_data', save_path=correlation_matrices_path)
target_corr.append(df_corr)

tc_feats = [tc for tc in target_corr]  # Lista de Index
tc_colna = [os.path.splitext(f)[0] for f in file_names]
tc_colna.append('training_data')  # Añadimos TRAINING_DATA

# Construir un DataFrame vacío con todas las features como filas y datasets como columnas
all_features = sorted(set().union(*tc_feats))  # Unión de todas las features
ranking_df = pd.DataFrame(index=all_features, columns=tc_colna)

# Llenar ranking_df
for col_name, feat_index in zip(tc_colna, tc_feats):
    for rank, feature in enumerate(feat_index):
        ranking_df.loc[feature, col_name] = rank + 1  # Ranking empieza en 1

# Convertir a int
ranking_df = ranking_df.astype(int)
ranking_df = ranking_df.sort_values(by='training_data', ascending=True)

# Plot general ranking
plt.figure(figsize=(12, 10))
sns.heatmap(ranking_df, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Ranking (1=High Correlation)'})
plt.title("Top Features Ranked by Correlation with the Target Variable")
plt.ylabel("Feature")
plt.xlabel("Data set transformations")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(correlation_matrices_path, "correlation_matrix_pearson_general_ranking_top.png"),
            dpi=300, bbox_inches='tight')
plt.close()

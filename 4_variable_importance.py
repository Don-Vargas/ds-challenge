import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils.storage import path_validate


def variable_importances(ax1, ax2, ax3, df, output_prefix, test_size, n_estimators, random_state):
    y = df['target']
    X = df.drop(columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    top_n = 20
    top_features = feature_importance_df.iloc[:top_n][::-1]
    ax1.barh(top_features['Feature'], top_features['Importance'])
    ax1.set_title(f'{output_prefix} - Top 20')
    ax1.set_xlabel("Importance")

    ordered_features = feature_importance_df['Feature'].tolist()

    num_features_list = []
    mae_list = []
    rmse_list = []

    for i in range(1, len(ordered_features) + 1):
        selected_features = ordered_features[:i]

        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train_subset, y_train)
        y_pred = rf.predict(X_test_subset)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        num_features_list.append(i)
        mae_list.append(mae)
        rmse_list.append(rmse)

    ax2.plot(num_features_list, mae_list, marker='o')
    ax2.set_title(f'{output_prefix} - MAE')
    ax2.set_xlabel('Dimensionality')
    ax2.set_ylabel('MAE')

    ax3.plot(num_features_list, rmse_list, marker='o', color='orange')
    ax3.set_title(f'{output_prefix} - RMSE')
    ax3.set_xlabel('Dimensionality')
    ax3.set_ylabel('RMSE')


importance_path = 'data/variable_importance/'
path_validate(importance_path)

folder_path = 'data/data_for_models/transformed_data/'
file_names = os.listdir(folder_path)
file_names = [f for f in file_names if f.endswith('.csv')]
n_files = len(file_names)

# Crear una figura con 3 filas (una por cada tipo de gráfico) y n columnas (una por cada archivo)
fig, axes = plt.subplots(3, n_files, figsize=(5 * n_files, 15))

# Add a global title for the entire figure
fig.suptitle('Variable importance and Metrics by model', fontsize=20)

# Optional: Adjust layout to make space for the title
fig.tight_layout(pad=5.0, rect=[0, 0, 1, 1])

# Si solo hay un archivo, axes será 1D, forzar a 2D
if n_files == 1:
    axes = axes.reshape(3, 1)

for i, fi_na in enumerate(file_names):
    df = pd.read_csv(f'{folder_path}{fi_na}')
    output_prefix = fi_na.split('.')[0]

    # Extraer los 3 ejes correspondientes a esta columna
    ax1, ax2, ax3 = axes[0, i], axes[1, i], axes[2, i]
    variable_importances(ax1, ax2, ax3, df, output_prefix, test_size=0.3, n_estimators=100, random_state=42)

# Guardar figura completa
plt.savefig(f'{importance_path}all_plots_grid.png', dpi=300, bbox_inches='tight')
plt.close()

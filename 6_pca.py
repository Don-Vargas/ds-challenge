import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.storage import path_validate, save_pickle


def select_pca_components(df, output_prefix, variance_threshold=0.80):
    """
    Aplica Análisis de Componentes Principales (PCA) a un DataFrame para reducir dimensionalidad,
    seleccionando el número mínimo de componentes necesarias para explicar al menos un porcentaje
    especificado de la varianza total. También guarda una gráfica de varianza acumulada y exporta 
    los datos transformados.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene las variables independientes y una columna llamada 'target'.
    output_prefix : str
        Prefijo que se usará para nombrar los archivos de salida (modelo, gráfica, CSV).
    variance_threshold : float, opcional (por defecto=0.80)
        Porcentaje mínimo de varianza explicada deseado (entre 0 y 1).

    Retorna:
    --------
    df_pca : pd.DataFrame
        DataFrame con las componentes principales seleccionadas y la columna 'target'.
    pca_model : sklearn.decomposition.PCA
        Modelo PCA ajustado.
    n_selected : int
        Número de componentes seleccionadas.
    cumulative_variance : np.ndarray
        Varianza acumulada explicada por las componentes principales.
    fig : matplotlib.figure.Figure
        Figura de la gráfica generada.
    """
    y = df['target']
    X = df.drop(columns=['target'])

    # Convertir a numpy array si es necesario
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    # Ajustar PCA
    pca = PCA()
    X_pca_full = pca.fit_transform(X_arr)

    # Guardar el modelo PCA
    save_pickle(pca, f'data/pca_models/pca_model_{output_prefix}.pkl')

    # Varianza explicada acumulada
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    # Seleccionar número de componentes
    n_selected = np.argmax(cumulative >= variance_threshold) + 1

    # Crear figura (no la mostramos aquí)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker='o', linestyle='--', label='Varianza acumulada')
    ax.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{int(variance_threshold*100)}% varianza')
    ax.axvline(x=n_selected, color='g', linestyle='--', label=f'{n_selected} componentes')
    ax.set_title(f'PCA - {output_prefix}')
    ax.set_xlabel('Número de componentes')
    ax.set_ylabel('Varianza acumulada')
    ax.set_xticks(range(1, len(cumulative) + 1))
    ax.grid(True)
    ax.legend()

    # Transformar los datos
    X_pca_sel = X_pca_full[:, :n_selected]
    col_names = [f'PC_{i+1}' for i in range(n_selected)]
    df_pca = pd.DataFrame(X_pca_sel, columns=col_names)
    df_pca['target'] = y

    # Guardar CSV
    pca_data_path = 'data/data_for_models/pca_data/'
    path_validate(pca_data_path)
    df_pca.to_csv(f'{pca_data_path}pca_{output_prefix}.csv', index=False)

    return df_pca, pca, n_selected, cumulative, fig

# Ejecutar PCA sobre múltiples archivos y mostrar gráficas en un solo grid
def run_pca_on_folder(folder_path, show_grid=True):
    file_names = sorted(os.listdir(folder_path))
    figs = []

    for file in file_names:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            output_prefix = os.path.splitext(file)[0]
            _, _, _, _, fig = select_pca_components(df, output_prefix, variance_threshold=0.80)
            figs.append((output_prefix, fig))

    if show_grid and figs:
        n = len(figs)
        cols = 3
        rows = (n + cols - 1) // cols
        fig_grid, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten()
        # Add a global title for the entire figure
        fig_grid.suptitle('PCA by transformation', fontsize=20)

        # Optional: Adjust layout to make space for the title
        fig_grid.tight_layout(pad=5.0, rect=[0, 0, 1, 1])

        for ax, (title, fig_item) in zip(axes, figs):
            # Copiar el contenido de cada figura al grid
            src_ax = fig_item.axes[0]
            for line in src_ax.lines:
                ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), linestyle=line.get_linestyle(), marker='o' if line.get_marker() else '')
            for hline in src_ax.lines:
                if hline.get_linestyle() == '--' and hline.get_color() == 'r':
                    ax.axhline(y=hline.get_ydata()[0], color='r', linestyle='--')
                if hline.get_linestyle() == '--' and hline.get_color() == 'g':
                    ax.axvline(x=hline.get_xdata()[0], color='g', linestyle='--')

            ax.set_title(f'PCA - {title}')
            ax.set_xlabel('Componentes')
            ax.set_ylabel('Varianza acumulada')
            ax.grid(True)
            ax.legend()

        # Desactivar subplots vacíos
        for ax in axes[len(figs):]:
            ax.axis('off')
        
        pca_path = 'data/pca/'
        path_validate(pca_path)
        plt.savefig(f"{pca_path}all_pca_grid.png", dpi=300, bbox_inches='tight')
        plt.close()

# Ejecutar todo
run_pca_on_folder('data/data_for_models/transformed_data/', show_grid=True)

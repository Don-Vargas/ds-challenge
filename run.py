
import subprocess
from pathlib import Path
import re
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DS_challenge_wizeline/'))


py_files = [f for f in os.listdir(parent_dir) if f.endswith('.py')]

def sort_files(file_list, order="asc"):
    def extract_number(filename):
        parts = filename.split("_", 1)
        try:
            return int(parts[0])  # Si empieza con número
        except ValueError:
            return float("inf")   # Si no tiene número, lo manda al final

    reverse = True if order == "desc" else False
    return sorted(file_list, key=extract_number, reverse=reverse)

# Ejemplos de uso
asc_files = sort_files(py_files, order="asc")
print(asc_files)

file_names = ['1_data_split.py', 
             '2_transform.py', 
             '3_eda.py', 
             '4_correlation_compute.py', 
             '5_correlation_visualization.py', 
             '6_kde_boxplot_visualization.py', 
             '7_variable_importance_compute.py', 
             '8_variable_importance_visualization.py', 
             '9_high_importance_dataset.py', 
             '10_pca_compute.py', 
             '11_pca_visualization.py', 
             '12_training_models.py']

def run_files(file_names):
    for f in file_names:
        print(f"🔹 Ejecutando {f}...")
        subprocess.run(["python3", f], check=True)

run_files(file_names)

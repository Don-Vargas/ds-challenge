
import subprocess
from pathlib import Path
import re
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DS-CHALLENGE/'))
print(parent_dir)

py_files = [f for f in os.listdir(parent_dir) if f.endswith('.py')]

print("Python scripts in parent directory:")
for file in py_files:
    print(file)
'''
def extract_number(filename):
    """Extrae el número inicial del nombre del archivo."""
    match = re.match(r'^(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# Ordenar la lista por número inicial
scripts_sorted = sorted(py_files, key=extract_number)

# Directorio donde están los scripts (ajustar si es necesario)
parent_dir = Path(__file__).resolve().parent.parent

for script_name in scripts_sorted:
    script_path = parent_dir / script_name
    print(f"Ejecutando: {script_name}")
    
    if not script_path.exists():
        print(f"Archivo no encontrado: {script_path}")
        break  # Si falta archivo, detiene la ejecución

    try:
        subprocess.run(['python', str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando {script_name}. Código de salida: {e.returncode}")
        print("Deteniendo ejecución.")
        break
'''
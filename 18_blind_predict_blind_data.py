'''
XGBoost models sorted by rmse:

                                           model_key         model_name                    dataset_name      rmse
17            xgboost_training_data_minmax_important            xgboost  training_data_minmax_important  0.068018
29             xgboost_training_data_minmax_original            xgboost   training_data_minmax_original  0.074489
0       random_forest_training_data_minmax_important      random_forest  training_data_minmax_important  0.078783
4        random_forest_training_data_minmax_original      random_forest   training_data_minmax_original  0.093290
15  linear_regression_training_data_minmax_important  linear_regression  training_data_minmax_important  0.096240
'''

import pandas as pd
from utils.registry import load_trained_registry
from utils.storage import load_pickle, path_validate

# 1. Cargar el registro entrenado
trained_registry = load_trained_registry()

# 2. Dataset a usar
dataset_path = 'data/data_for_models/blind_data_sets/important_blind_data_minmax.csv'
X_blind = pd.read_csv(dataset_path)

# 3. Obtener el modelo específico por nombre
xgb_model_info = trained_registry.get("xgboost_training_data_minmax_important")

if xgb_model_info:
    print("Modelo encontrado")
    print("Ruta del pickle:", xgb_model_info["model_pickle_path"])
    
    # 4. Cargar el modelo entrenado desde el pickle usando tu función
    model = load_pickle(xgb_model_info["model_pickle_path"])
    
    # 5. Hacer predicciones
    predictions = model.predict(X_blind).reshape(-1, 1)  # reshape necesario para inverse_transform
    
    # 6. Reverse transform si existe scaler
    scaler_path = 'data/scalers/scaler_important_minmax.pkl'
    try:
        scaler = load_pickle(scaler_path)
        predictions = scaler.inverse_transform(predictions)
        print("Predicciones reescaladas a la escala original")
    except FileNotFoundError:
        print(f"No se encontró scaler en {scaler_path}, se guardan predicciones tal cual")
    
    # 7. Guardar resultados
    output_path = "data/data_for_models/blind_predictions/blind_predictions_xgboost.csv"
    path_validate(output_path)
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_path, index=False)
    print(f"Predicciones generadas y guardadas en {output_path}")
else:
    print("No se encontró el modelo en el registry")






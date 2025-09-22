import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils.storage import path_validate
from utils.registry import load_registry, add_importance_path_to_registry

def variable_importances(df, test_size=0.3, n_estimators=100, random_state=42):
    y = df['target']
    X = df.drop(columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    ordered_features = feature_importance_df['Feature'].tolist()

    results = []
    for i in range(1, len(ordered_features) + 1):
        selected_features = ordered_features[:i]
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train_subset, y_train)
        y_pred = rf.predict(X_test_subset)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        results.append({
            'num_features': i,
            'mae': mae,
            'rmse': rmse
        })

    metrics_df = pd.DataFrame(results)

    return feature_importance_df, metrics_df

# Paths
importance_path = 'data/variable_importance/'
path_validate(importance_path)

# Load registry
registry = load_registry()

# Loop through registry items
all_results = {}

for prefix, info in registry.items():
    output_csv_path = info.get("output_csv_path")

    if not output_csv_path or not os.path.isfile(output_csv_path):
        print(f"[WARNING] Skipping '{prefix}' — missing or invalid output_csv_path.")
        continue

    print(f"[INFO] Processing '{prefix}'...")
    df = pd.read_csv(output_csv_path)

    # Run variable importance analysis
    feature_importance_df, metrics_df = variable_importances(
        df, test_size=0.3, n_estimators=100, random_state=42
    )

    # Save results
    importance_file = os.path.join(importance_path, f'feature_importance_{prefix}.csv')
    metrics_file = os.path.join(importance_path, f'metrics_{prefix}.csv')
    add_importance_path_to_registry(prefix, importance_paths=[importance_file,metrics_file])

    feature_importance_df.to_csv(importance_file, index=False)
    metrics_df.to_csv(metrics_file, index=False)

    # Store results in memory if needed
    all_results[prefix] = {
        'feature_importance': feature_importance_df,
        'metrics': metrics_df
    }

print(f"[INFO] Finished processing {len(all_results)} datasets.")

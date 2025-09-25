import pandas as pd
from sklearn.model_selection import train_test_split

from utils.path_manager import initialize_data_paths
from config.paths import (TRAINING_DATA_TRANSFORMED_DIR, 
                          TEST_DATA_TRANSFORMED_DIR, 
                          BLIND_DATA_TRANSFORMED_DIR, 
                          ORIGINAL_DATA_FILE, 
                          BLIND_DATA_FILE)


def split_and_save_datasets(test_size=0.2, random_state=42):
    '''
    split original data
    '''
    initialize_data_paths()
    df      = pd.read_csv(ORIGINAL_DATA_FILE)
    blind   = pd.read_csv(BLIND_DATA_FILE)
    
    # Split into train/test (80/20)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save to respective folders
    train_df.to_csv(f'{TRAINING_DATA_TRANSFORMED_DIR}train.csv', index=False)
    test_df.to_csv(f'{TEST_DATA_TRANSFORMED_DIR}test.csv', index=False)
    blind.to_csv(f'{BLIND_DATA_TRANSFORMED_DIR}blind.csv', index=False)

# ----------------------------
# MAIN EXECUTION BLOCK
# ----------------------------
if __name__ == "__main__":
    split_and_save_datasets(test_size=0.2, random_state=42)
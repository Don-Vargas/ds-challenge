# Base data folders
BASE_DIR    = 'data'
ORIGIN_DIR   = f'{BASE_DIR}/A_original_data'
TRAIN_DIR   = f'{BASE_DIR}/B_training_data'
TEST_DIR    = f'{BASE_DIR}/C_test_data'
BLIND_DIR   = f'{BASE_DIR}/D_blind_data'

# File/directory paths
TRAINING_DATA_TRANSFORMED_DIR   = f'{TRAIN_DIR}/datasets/data_transformed/'
TEST_DATA_TRANSFORMED_DIR   = f'{TEST_DIR}/datasets/data_transformed/'
BLIND_DATA_TRANSFORMED_DIR  = f'{BLIND_DIR}/datasets/data_transformed/'

# Transformed Model/directory paths
TRAINING_MODEL_TRANSFORMED_DIR   = f'{TRAIN_DIR}/models/data_transformed/'
TEST_MODEL_TRANSFORMED_DIR       = f'{TEST_DIR}/models/data_transformed/'
BLIND_MODEL_TRANSFORMED_DIR      = f'{BLIND_DIR}/models/data_transformed/'

# Original data path
ORIGINAL_DATA_FILE  = f'{ORIGIN_DIR}/training_data.csv'
BLIND_DATA_FILE     = f'{ORIGIN_DIR}/blind_test_data.csv'
TRAIN_SPLIT_RAW_FILE= f'{TRAINING_DATA_TRANSFORMED_DIR}/train.csv'
TEST_SPLIT_RAW_FILE = f'{TEST_DATA_TRANSFORMED_DIR}/test.csv'
BLIND_RAW_FILE      = f'{BLIND_DATA_TRANSFORMED_DIR}/blind.csv'

# Base data folders
BASE_REGISTRY = 'registry'

# Centralized path dictionary
data_paths = {
    'train': {
        'base_data': TRAIN_DIR,
        'transformed_data': TRAINING_DATA_TRANSFORMED_DIR,
        'transformed_model': TRAINING_MODEL_TRANSFORMED_DIR
    },
    'test': {
        'base_data': TEST_DIR,
        'transformed_data': TEST_DATA_TRANSFORMED_DIR,
        'transformed_model': TEST_MODEL_TRANSFORMED_DIR
    },
    'blind': {
        'base_data': BLIND_DIR,
        'transformed_data': BLIND_DATA_TRANSFORMED_DIR,
        'transformed_model': BLIND_MODEL_TRANSFORMED_DIR
    },
    'registry':{
        'base registry': BASE_REGISTRY
    }
}

# Registry file paths
REGISTRY_FILE = f'{BASE_REGISTRY}/model_registry.json'
TRAINED_REGISTRY_FILE = f'{BASE_REGISTRY}/trained_model_registry.json'

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
ORIGINAL_DATA_FILE      = f'{ORIGIN_DIR}/training_data.csv'
BLIND_DATA_FILE         = f'{ORIGIN_DIR}/blind_test_data.csv'
TRAINING_SPLIT_RAW_FILE = f'{TRAINING_DATA_TRANSFORMED_DIR}/train.csv'
TEST_SPLIT_RAW_FILE     = f'{TEST_DATA_TRANSFORMED_DIR}/test.csv'
BLIND_RAW_FILE          = f'{BLIND_DATA_TRANSFORMED_DIR}/blind.csv'

# Analisys path
EDA_TRAINING_DATA_DIR   = f'{TRAIN_DIR}/analisys/eda/'
CORR_TRAINING_DATA_DIR  = f'{TRAIN_DIR}/analisys/correlation_matrices/'
VAR_IMPORTANCE_DATA_DIR = f'{TRAIN_DIR}/analisys/variable_importance/'
CORR_FIGS_DIR           = f'{TRAIN_DIR}/analisys/figures/correlation/'
KDE_FIGS_DIR            = f'{TRAIN_DIR}/analisys/figures/kde/'
IMPORTANCE_FIGS_DIR     = f'{TRAIN_DIR}/analisys/figures/importance/'
PCA_FIGS_DIR            = f'{TRAIN_DIR}/analisys/figures/pca/'

# Important variables datasets
TRAINING_DATA_IMPORTANT_DIR = f'{TRAIN_DIR}/datasets/data_important/'
TEST_DATA_IMPORTANT_DIR     = f'{TEST_DIR}/datasets/data_important/'
BLIND_DATA_IMPORTANT_DIR    = f'{BLIND_DIR}/datasets/data_important/'

# PCA datasets
TRAINING_DATA_PCA_DIR   = f'{TRAIN_DIR}/datasets/data_pca/'
TEST_DATA_PCA_DIR       = f'{TEST_DIR}/datasets/data_pca/'
BLIND_DATA_PCA_DIR      = f'{BLIND_DIR}/datasets/data_pca/'

# PCA Model/directory paths
TRAINING_MODEL_PCA_DIR   = f'{TRAIN_DIR}/models/data_pca/'
TEST_MODEL_PCA_DIR       = f'{TEST_DIR}/models/data_pca/'
BLIND_MODEL_PCA_DIR      = f'{BLIND_DIR}/models/data_pca/'

# Base data folders
BASE_REGISTRY = 'registry'

# Centralized path dictionary
data_paths = {
    'train': {
        'base_data': TRAIN_DIR,
        'transformed_data': TRAINING_DATA_TRANSFORMED_DIR,
        'transformed_model': TRAINING_MODEL_TRANSFORMED_DIR,
        'eda': EDA_TRAINING_DATA_DIR,
        'corr': CORR_TRAINING_DATA_DIR,
        'importance':VAR_IMPORTANCE_DATA_DIR,
        'figures_corr': CORR_FIGS_DIR,
        'figures_kde': KDE_FIGS_DIR,
        'figures_imp': IMPORTANCE_FIGS_DIR,
        'figures_pca': PCA_FIGS_DIR,
        'important': TRAINING_DATA_IMPORTANT_DIR,
        'pca_data': TRAINING_DATA_PCA_DIR,
        'pca_model': TRAINING_MODEL_PCA_DIR
    },
    'test': {
        'base_data': TEST_DIR,
        'transformed_data': TEST_DATA_TRANSFORMED_DIR,
        'transformed_model': TEST_MODEL_TRANSFORMED_DIR,
        'important': TEST_DATA_IMPORTANT_DIR,
        'pca_data': TEST_DATA_PCA_DIR,
        'pca_model': TEST_MODEL_PCA_DIR
    },
    'blind': {
        'base_data': BLIND_DIR,
        'transformed_data': BLIND_DATA_TRANSFORMED_DIR,
        'transformed_model': BLIND_MODEL_TRANSFORMED_DIR,
        'important': BLIND_DATA_IMPORTANT_DIR,
        'pca_data': BLIND_DATA_PCA_DIR,
        'pca_model': BLIND_MODEL_PCA_DIR
    },
    'registry':{
        'base registry': BASE_REGISTRY
    }
}

# Registry file paths
REGISTRY_FILE = f'{BASE_REGISTRY}/model_registry.json'
TRAINED_REGISTRY_FILE = f'{BASE_REGISTRY}/trained_model_registry.json'

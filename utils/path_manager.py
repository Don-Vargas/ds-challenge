from config.paths import data_paths
from utils.storage import path_validate

def validate_all_paths(paths_dict):
    for category in paths_dict.values():
        for path in category.values():
            path_validate(path)

def initialize_data_paths():
    validate_all_paths(data_paths)


import pickle
import os

VALIDATION_DATA_FILE = 'preprocess_validation.p'


def get_validation_file_path(path):
    """
    Return the checkpoint bin data path.
    :param path: object root
    :return: bin-data path
    """
    return os.path.join(path, VALIDATION_DATA_FILE)


def load_source(path):
    """
    Load serialized data.
    :param path: source path
    :return: valid_features, valid_labels
    """
    # Load the Preprocessed Validation data
    file = None
    try:
        file = open(path, mode='rb')
        valid_features, valid_labels = pickle.load(file)
        return valid_features, valid_labels
    finally:
        if file is not None:
            file.close()

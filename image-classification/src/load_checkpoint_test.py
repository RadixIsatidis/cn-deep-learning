import unittest

from src.load_checkpoint import VALIDATION_DATA_FILE, get_validation_file_path, load_source


class TestLoadCheckpoint(unittest.TestCase):

    def test_getValidationFilePath(self):
        path = 'c:\\foo\\bar\\'
        resolved_path = get_validation_file_path(path)
        self.assertEqual(resolved_path, "c:\\foo\\bar\\" + VALIDATION_DATA_FILE)

    def test_loadSource(self):
        path = '.'
        path = get_validation_file_path(path)
        valid_features, valid_labels = load_source(path)
        self.assertIsNotNone(valid_labels)
        self.assertIsNotNone(valid_labels)

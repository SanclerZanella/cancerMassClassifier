import unittest
from sklearn.utils import Bunch
from main.data_processing.diagnosedatasetfetcher import DiagnoseDatasetFetcher
from test.main.utils import DATASET_DESCRIPTION


class DiagnoseDatasetFetcherTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This runs once before all tests
        cls.target_class = DiagnoseDatasetFetcher()

    @classmethod
    def tearDownClass(cls):
        # This runs once after all tests
        del cls.target_class

    def test_dataset_type(self):
        # Check dataset type
        self.assertEqual(type(self.target_class.dataset), Bunch)

    def test_dataset_description(self):
        # Check dataset description type
        self.assertEqual(type(self.target_class.dataset_description()), str)

        # Check dataset content
        self.assertEqual(self.target_class.dataset_description(), DATASET_DESCRIPTION)

import unittest
from sklearn.utils import Bunch
from main.data_processing.diagnosedatasetfetcher import DiagnoseDatasetFetcher
from test.main.utils import DATASET_DESCRIPTION


class DiagnoseDatasetFetcherTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.target_class = DiagnoseDatasetFetcher()

    def tearDown(self) -> None:
        del self.target_class

    def test_dataset_type(self):
        # Check dataset type
        self.assertEqual(type(self.target_class.dataset), Bunch)

    def test_dataset_description(self):
        # Check dataset description type
        self.assertEqual(type(self.target_class.dataset_description()), str)

        # Check dataset content
        self.assertEqual(self.target_class.dataset_description(), DATASET_DESCRIPTION)

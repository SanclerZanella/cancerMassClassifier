import unittest
from unittest.mock import Mock
from main.data_processing.models.data_classifier_LR_model import LogisticRegressionModel


class LogisticRegressionModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_dataset = Mock()

        # Mocking the data expected by the DataFrame constructor
        self.mock_dataset.dataset.data = self.mock_dataset.dataset.data = [
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],
        ]
        self.mock_dataset.dataset.feature_names = ['col1', 'col2']
        self.mock_dataset.dataset.target = [0, 1, 0, 1]

        self.target_class = LogisticRegressionModel(self.mock_dataset)

    def tearDown(self) -> None:
        del self.mock_dataset
        del self.target_class

    def test_model_score(self):
        score = self.target_class.model_score()
        self.assertEqual(score, 0.0)

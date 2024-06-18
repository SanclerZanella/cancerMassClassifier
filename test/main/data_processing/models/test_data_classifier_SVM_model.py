import unittest
from unittest.mock import Mock
from numpy import ndarray
from main.data_processing.models.data_classifier_SVM_model import DataClassifier


class DataClassifierTestCase(unittest.TestCase):
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

        self.target_class = DataClassifier(self.mock_dataset)

    def tearDown(self) -> None:
        del self.mock_dataset
        del self.target_class

    def test_model_prediction(self):
        model_prediction = self.target_class.model_prediction()

        # Check prediction type
        self.assertEqual(type(model_prediction), ndarray)

        # Check prediction length
        self.assertEqual(len(model_prediction), len(self.target_class.Y_test))

    def test_prediction_accuracy(self):
        model_accuracy = self.target_class.prediction_accuracy()

        # Check accuracy score
        self.assertIsInstance(model_accuracy, float)
        self.assertGreaterEqual(model_accuracy, 0.0)
        self.assertLessEqual(model_accuracy, 100.0)

    def test_prediction_precision(self):
        model_precision = self.target_class.prediction_precision()

        # Check precision score
        self.assertIsInstance(model_precision, float)
        self.assertGreaterEqual(model_precision, 0.0)
        self.assertLessEqual(model_precision, 100.0)

    def test_prediction_recall(self):
        model_recall = self.target_class.prediction_recall()

        # Check recall score
        self.assertIsInstance(model_recall, float)
        self.assertGreaterEqual(model_recall, 0.0)
        self.assertLessEqual(model_recall, 100.0)


if __name__ == '__main__':
    unittest.main()

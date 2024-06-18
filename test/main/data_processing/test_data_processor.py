import unittest
from unittest.mock import Mock

from pandas import DataFrame, testing
from sklearn.model_selection import train_test_split

from main.data_processing.data_processor import ProcessDataset


class ProcessDatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_dataset = Mock()
        self.target_class = ProcessDataset

    def tearDown(self) -> None:
        del self.mock_dataset
        del self.target_class

    def test_dataframe_generator(self):
        # Mocking the data expected by the DataFrame constructor
        self.mock_dataset.dataset.data = self.mock_dataset.dataset.data = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        self.mock_dataset.dataset.feature_names = ['col1', 'col2']

        t_class = self.target_class(self.mock_dataset)
        dataframe_generator = t_class.dataframe_generator()

        # Check if the output is indeed a DataFrame
        self.assertEqual(type(dataframe_generator), DataFrame)

        expected_df = DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        # Check if the output is correct
        testing.assert_frame_equal(dataframe_generator, expected_df)

    def test_split_dataset(self):
        # Mocking the data expected by the DataFrame constructor
        self.mock_dataset.dataset.data = self.mock_dataset.dataset.data = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        self.mock_dataset.dataset.feature_names = ['col1', 'col2']
        self.mock_dataset.dataset.target = [0, 1, 0]

        t_class = self.target_class(self.mock_dataset)
        splitted_dataset = t_class.split_dataset()

        # Check output type
        self.assertEqual(type(splitted_dataset), tuple)

        # Check output size
        self.assertEqual(len(splitted_dataset), 4)

        X_train, X_test, Y_train, Y_test = splitted_dataset

        # Check if the lengths of the splits are correct
        self.assertEqual(len(X_train) + len(X_test), len(self.mock_dataset.dataset.data))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.mock_dataset.dataset.target))

        # heck the content of the splits
        expected_X_train, expected_X_test, expected_Y_train, expected_Y_test = train_test_split(
            self.mock_dataset.dataset.data, self.mock_dataset.dataset.target, random_state=0
        )

        testing.assert_frame_equal(DataFrame(X_train, columns=self.mock_dataset.dataset.feature_names),
                                   DataFrame(expected_X_train, columns=self.mock_dataset.dataset.feature_names))

        testing.assert_frame_equal(DataFrame(X_test, columns=self.mock_dataset.dataset.feature_names),
                                   DataFrame(expected_X_test, columns=self.mock_dataset.dataset.feature_names))

        self.assertEqual(Y_train, expected_Y_train)
        self.assertEqual(Y_test, expected_Y_test)


if __name__ == '__main__':
    unittest.main()

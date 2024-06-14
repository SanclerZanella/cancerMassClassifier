import unittest
from unittest.mock import Mock

from pandas import DataFrame, testing
from sklearn.model_selection import train_test_split

from main.data_processing.data_processor import ProcessDataset


class DiagnoseDatasetFetcherTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This runs once before all tests
        cls.mock = Mock()
        cls.target_class = ProcessDataset

    @classmethod
    def tearDownClass(cls):
        # This runs once after all tests
        del cls.mock
        del cls.target_class

    def test_dataframe_generator(self):
        # Mocking the data expected by the DataFrame constructor
        self.mock.dataset.data = self.mock.dataset.data = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        self.mock.dataset.feature_names = ['col1', 'col2']

        t_class = self.target_class(self.mock)
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
        self.mock.dataset.data = self.mock.dataset.data = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        self.mock.dataset.feature_names = ['col1', 'col2']
        self.mock.dataset.target = [0, 1, 0]

        t_class = self.target_class(self.mock)
        splitted_dataset = t_class.split_dataset()

        # Check output type
        self.assertEqual(type(splitted_dataset), tuple)

        # Check output size
        self.assertEqual(len(splitted_dataset), 4)

        X_train, X_test, Y_train, Y_test = splitted_dataset

        # Check if the lengths of the splits are correct
        self.assertEqual(len(X_train) + len(X_test), len(self.mock.dataset.data))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.mock.dataset.target))

        # heck the content of the splits
        expected_X_train, expected_X_test, expected_Y_train, expected_Y_test = train_test_split(
            self.mock.dataset.data, self.mock.dataset.target, random_state=0
        )

        testing.assert_frame_equal(DataFrame(X_train, columns=self.mock.dataset.feature_names),
                                   DataFrame(expected_X_train, columns=self.mock.dataset.feature_names))

        testing.assert_frame_equal(DataFrame(X_test, columns=self.mock.dataset.feature_names),
                                   DataFrame(expected_X_test, columns=self.mock.dataset.feature_names))

        self.assertEqual(Y_train, expected_Y_train)
        self.assertEqual(Y_test, expected_Y_test)


if __name__ == '__main__':
    unittest.main()

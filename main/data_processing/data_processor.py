import pandas
from main.data_processing.diagnosedatasetfetcher import DiagnoseDatasetFetcher
from sklearn.model_selection import train_test_split


class ProcessDataset:
    def __init__(self, diagnose_dataset: DiagnoseDatasetFetcher):
        self.data = diagnose_dataset.dataset.data
        self.target = diagnose_dataset.dataset.target
        self.column_titles = diagnose_dataset.dataset.feature_names

    def dataframe_generator(self):
        """
            Convert the dataset into a pandas DataFrame
        """

        return pandas.DataFrame(
            self.data,
            columns=self.column_titles
        )

    def split_dataset(self):
        """
            Split dataset for training and testing

            X_train, X_test: These variables represents the features to be trained and tested.
            Y_train, Y_test: These variables represents the dataset labels.
        """
        (X_train, X_test,
         Y_train, Y_test) = train_test_split(self.data,
                                                self.target,
                                                random_state=0)
        return X_train, X_test, Y_train, Y_test

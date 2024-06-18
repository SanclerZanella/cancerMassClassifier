from sklearn import svm, metrics
from main.data_processing.data_processor import ProcessDataset
from main.data_processing.diagnosedatasetfetcher import DiagnoseDatasetFetcher


class DataClassifier:
    def __init__(self, dataset: DiagnoseDatasetFetcher):
        self.processed_data = ProcessDataset(dataset)
        (self.X_train, self.X_test,
         self.Y_train, self.Y_test) = self.processed_data.split_dataset()
        self.model = svm.SVC(kernel="linear")  # Support vector classifier

    def model_prediction(self):
        self.model.fit(self.X_train, self.Y_train)
        return self.model.predict(self.X_test)

    def prediction_accuracy(self):
        #  Compare models's prediction with the actual data, to check models's prediction accuracy
        return metrics.accuracy_score(self.Y_test, self.model_prediction()) * 100

    def prediction_precision(self):
        #  Compare models's prediction with the actual data, to check models's prediction precision
        return metrics.precision_score(self.Y_test, self.model_prediction()) * 100

    def prediction_recall(self):
        #  Compare models's prediction with the actual data, to check models's prediction recall
        return metrics.recall_score(self.Y_test, self.model_prediction()) * 100

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from main.data_processing.data_processor import ProcessDataset
from main.data_processing.diagnosedatasetfetcher import DiagnoseDatasetFetcher


class LogisticRegressionModel:
    def __init__(self, dataset: DiagnoseDatasetFetcher):
        self.processed_data = ProcessDataset(dataset)
        (self.X_train, self.X_test,
         self.Y_train, self.Y_test) = self.processed_data.split_dataset()
        self.scaler = StandardScaler()
        self.logistic_model = LogisticRegression()

    def model_score(self):
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        self.logistic_model.fit(X_train_scaled, self.Y_train)

        return self.logistic_model.score(X_test_scaled, self.Y_test)

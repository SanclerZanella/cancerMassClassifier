from sklearn.datasets import load_breast_cancer


class DiagnoseDatasetFetcher:
    def __init__(self):
        self.dataset = load_breast_cancer()

    def dataset_description(self) -> str:
        return str(self.dataset.DESCR).strip()

from dataset.base import BaseDataset
from dataset.utils import apply_pca_if_needed

class BankLoanDataset(BaseDataset):
    """
    House price prediction. boston_house_prices.csv
    """
    def __init__(self, file_path, test_split=0.2, generated_codes_path="./cache/generated_codes.json"):
        super().__init__(file_path, test_split, generated_codes_path)

    def preprocessing(self, data, query=None):
        X = data.values
        X = apply_pca_if_needed(X, n_components=10)
        y = [-1] * len(data)
        return X, y

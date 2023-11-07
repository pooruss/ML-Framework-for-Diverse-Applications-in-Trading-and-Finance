from  dataset.base import BaseDataset

class CreditCardFraudDetection(BaseDataset):
    """
    Credit Card Fraud Detection. creditcard_2023.csv. 
    https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data
    """
    def __init__(self, file_path, test_split=0.2, generated_codes_path="./cache/generated_codes.json"):
        super().__init__(file_path, test_split, generated_codes_path)


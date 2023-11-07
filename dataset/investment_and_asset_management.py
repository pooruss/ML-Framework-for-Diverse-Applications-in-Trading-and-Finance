from dataset.base import BaseDataset
from dataset.utils import normalization, standardization

class HousePricePrediction(BaseDataset):
    """
    House price prediction. boston_house_prices.csv
    """
    def __init__(self, file_path, test_split=0.2, generated_codes_path="./cache/generated_codes.json"):
        super().__init__(file_path, test_split, generated_codes_path)
    
    def preprocessing(self, data, query=None):
        pass

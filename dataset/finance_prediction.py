from  dataset.base import BaseDataset

class StockPredictionDataset(BaseDataset):
    def __init__(self, file_path, test_split=0.2, generated_codes_path="./cache/generated_codes.json"):
        super().__init__(file_path, test_split, generated_codes_path)

    def preprocessing(self,  data, query=None):
        # Create the target variable - 1 if Close price is higher than Open price, otherwise 0
        data['Target'] = (data['Close'] > data['Open']).astype(int)
        # Drop the 'Date' and 'Adj Close' columns
        data = data.drop(['Date', 'Adj Close'], axis=1)
        # Split the data into features and target
        X = data.drop('Target', axis=1).values
        y = data['Target'].values
        return X, y
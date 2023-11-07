import pandas as pd
import numpy as np
from dataset.base import BaseDataset

class CustomerChurnPrediction(BaseDataset):
    """
    Credit Card Customer Churn Prediction. 
    """
    def __init__(self, file_path, test_split=0.2, generated_codes_path="./cache/generated_codes.json"):
        super().__init__(file_path, test_split, generated_codes_path)

    def preprocessing(self, data: pd.DataFrame, query=None):
        data.drop(columns=["RowNumber","CustomerId","Surname"], inplace = True)
        data = pd.get_dummies(data, columns=['Geography','Gender'], drop_first=True)
        data.loc[data['Geography_Germany'] == True, 'x1'] = 1
        data.loc[data['Geography_Germany'] == False, 'x1'] = 0
        data.loc[data['Geography_Spain'] == True, 'x2'] = 1
        data.loc[data['Geography_Spain'] == False, 'x2'] = 0
        data.loc[data['Gender_Male'] == True, 'x3'] = 1
        data.loc[data['Gender_Male'] == False, 'x3'] = 0

        data.drop(columns=["Gender_Male","Geography_Spain","Geography_Germany"], inplace = True)
        size_train = int(data.shape[0] * 0.8)
        xtrain = np.asarray(data.iloc[0:size_train].drop(columns=["Exited"]))
        ytrain = np.asarray(data.iloc[0:size_train]['Exited']).flatten()
        xtest = np.asarray(data.iloc[size_train:].drop(columns=["Exited"]))
        ytest = np.asarray(data.iloc[size_train:]['Exited']).flatten()
        return (xtrain, ytrain), (xtest, ytest)

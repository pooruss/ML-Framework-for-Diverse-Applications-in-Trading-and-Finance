import json
import pandas as pd
import yaml
import numpy as np
from abc import ABC, abstractmethod
np.random.seed(1)

class BaseAlgorithm(ABC):
    def __init__(self, config_file: str) -> None:
        self.config = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        return super().__str__()
    
    def save_params(self, file_path):
        params = {attr: getattr(self, attr) for attr in self.__dict__ if not attr.startswith('_')}
        for key in params:
            if isinstance(params[key], np.ndarray):
                params[key] = params[key].tolist()
        json.dump(params, open(file_path, "w"), indent=2)

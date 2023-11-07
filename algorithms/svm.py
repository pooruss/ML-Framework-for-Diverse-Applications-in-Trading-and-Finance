import numpy as np
from algorithms.base import BaseAlgorithm

# SimpleSVM class
class SimpleSVM(BaseAlgorithm):
    """
    A simple implementation of a Support Vector Machine using gradient descent.
    """
    def __init__(self, config_file):
        super().__init__(config_file)
        self.lr = self.config["learning_rate"]
        self.lambda_param = self.config["lambda_param"]
        self.n_iters = self.config["n_iters"]
        self.w = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize parameters to zero
        self.w = np.zeros(n_features)
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w)) >= 1
                if condition:
                    # If correctly classified, only regularize w
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If incorrectly classified, update w
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
    
    def predict(self, X):
        linear_output = np.dot(X, self.w)
        return np.sign(linear_output)
    
    def __str__(self) -> str:
        return "SimpleSVM"
    
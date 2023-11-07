import numpy as np
from algorithms.base import BaseAlgorithm

class LinearRegression(BaseAlgorithm):
    """
    A simple implementation of Linear Regression using gradient descent.
    """
    def __init__(self, config_file):
        super().__init__(config_file)
        self.lr = self.config.get("learning_rate", 0.01)
        self.n_iters = self.config.get("n_iters", 1000)
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
        
    def __str__(self):
        return f"LinearRegression(lr={self.lr}, n_iters={self.n_iters})"
    
import numpy as np
from algorithms.base import BaseAlgorithm

class PCA(BaseAlgorithm):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        # Assuming the configuration file contains the number of principal components to retain
        self.n_components = self.config.get('n_components', None)
        self.components_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort the eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the first n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_components]
        
    def predict(self, X):
        # Check if fit has been called
        if self.components_ is None:
            raise RuntimeError("The PCA algorithm has not been fitted yet.")
        
        # Project the data onto the principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def __str__(self) -> str:
        return f"PCA(n_components={self.n_components})"
    
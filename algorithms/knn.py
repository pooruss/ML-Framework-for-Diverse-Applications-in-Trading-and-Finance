import numpy as np
from algorithms.base import BaseAlgorithm
# Enhancing the KNN class to support different distance calculation methods

class KNN(BaseAlgorithm):
    def __init__(self, config_file: str, k: int = 3):
        super().__init__(config_file)
        self.k = k  # Number of neighbors
        self.distance_method = self.config.get('distance_method', 'euclidean').lower()
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Memorize the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Predict the label for each point in X
        predictions = []
        for point in X:
            predictions.append(self._predict_point(point))
        return predictions

    def _predict_point(self, point):
        # Find the k nearest neighbors using the specified distance method
        distances = self._calculate_distances(point)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

    def _calculate_distances(self, point):
        # Calculate distances based on the specified method
        if self.distance_method == 'manhattan':
            return np.sum(np.abs(self.X_train - point), axis=1)
        else:  # Default to Euclidean
            return np.linalg.norm(self.X_train - point, axis=1)

    def __str__(self):
        return f"KNN(k={self.k}, distance_method={self.distance_method})"

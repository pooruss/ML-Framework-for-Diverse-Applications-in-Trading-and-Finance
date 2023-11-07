import numpy as np
from algorithms.base import BaseAlgorithm

class DBSCAN(BaseAlgorithm):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.eps = self.config["eps"]
        self.min_samples = self.config["min_samples"]

    def find_all_neighbors(self, distances):
        neighbors = set()
        for i, distance in enumerate(distances):
            if distance <= self.eps:
                neighbors.add(i)
        return neighbors

    def find_core_points(self, dataset):
        core_points = set()
        distances = np.linalg.norm(dataset - dataset[:, np.newaxis], axis=2)

        for i in range(len(dataset)):
            neighbors = self.find_all_neighbors(distances[i])
            if len(neighbors) >= self.min_samples:
                core_points.add(i)
        
        return core_points

    def fix(self, X, y):
        pass

    def predict(self, X):
        labels = [-1] * len(X)
        cluster_label = 0
        core_points = self.find_core_points(X)
        distances = np.linalg.norm(X - X[:, np.newaxis], axis=2)

        for core_point in core_points:
            if labels[core_point] != -1:
                continue

            neighbors = self.find_all_neighbors(distances[core_point])
            labels[core_point] = cluster_label

            neighbors_copy = set(neighbors.copy())  # Create a copy for iteration
            for neighbor_i in neighbors_copy:
                if labels[neighbor_i] == -2:
                    labels[neighbor_i] = cluster_label
                if labels[neighbor_i] != -1:
                    continue
                labels[neighbor_i] = cluster_label
                new_neighbors = self.find_all_neighbors(distances[neighbor_i])
                if len(new_neighbors) >= self.min_samples:
                    neighbors.update(new_neighbors)

            cluster_label += 1
        return np.array(labels)

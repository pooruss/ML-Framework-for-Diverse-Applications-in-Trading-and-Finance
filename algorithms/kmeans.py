from algorithms.base import BaseAlgorithm
import numpy as np

class KMeans(BaseAlgorithm):
    """
    A simple implementation of the KMeans clustering algorithm.
    """
    def __init__(self, config_file):
        super().__init__(config_file)
        self.K = self.config.get("n_clusters", 3)
        self.max_iters = self.config.get("max_iters", 300)
        self.centroids = None
        
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iters):
            clusters = self._create_clusters(X)
            previous_centroids = self.centroids
            self.centroids = self._calculate_new_centroids(clusters, X)
            
            if self._is_converged(previous_centroids, self.centroids):
                break
        
        self.labels_ = self._get_cluster_labels(X, clusters)
        
    def predict(self, X):
        return self._closest_centroid(X)
    
    def _initialize_centroids(self, X):
        random_sample_idxs = np.random.choice(X.shape[0], self.K, replace=False)
        centroids = X[random_sample_idxs]
        return centroids
    
    def _closest_centroid(self, points):
        distances = np.sqrt(((points - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.K)]
        for idx, point in enumerate(X):
            centroid_idx = self._closest_centroid(point)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, X.shape[1]))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids
    
    def _is_converged(self, previous_centroids, new_centroids):
        distances = np.sqrt(((new_centroids - previous_centroids)**2).sum(axis=1))
        return np.all(distances < 1e-6)
    
    def _get_cluster_labels(self, X, clusters):
        labels = np.empty(X.shape[0])
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels
        
    def __str__(self):
        return f"KMeans(K={self.K}, max_iters={self.max_iters})"
    
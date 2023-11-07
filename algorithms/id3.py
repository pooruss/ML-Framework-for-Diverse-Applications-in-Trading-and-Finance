import numpy as np
from algorithms.base import BaseAlgorithm

class ID3(BaseAlgorithm):
    def __init__(self, config_file):
        super().__init__(config_file)
        # Tree structure initialized as None before fitting
        self.tree = None
        
    def fit(self, X, y):
        # Build the decision tree using the ID3 algorithm
        self.tree = self._build_tree(X, y)
    
    def predict(self, X):
        # Predict the class label for a given instance using the decision tree
        return np.array([self._predict_single(instance) for instance in X])
    
    def _calculate_entropy(self, y):
        # Calculate the empirical entropy (Shannon entropy) of the dataset
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def _build_tree(self, X, y):
        # Base case: if the dataset is pure (all instances have the same class), return the class
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 1:
            return unique_classes[0]
        
        # Base case: if the dataset is empty or there are no features left, return the most common class
        if X.shape[0] == 0 or X.shape[1] == 0:
            return unique_classes[np.argmax(counts)]
        
        # Choose the best feature to split on
        best_feature = self._choose_best_feature_to_split(X, y)
        
        # If no feature brings information gain, return the most common class
        if best_feature is None:
            return unique_classes[np.argmax(counts)]
        
        # Create the tree structure, starting with the root node (best feature)
        tree = {best_feature: {}}
        
        # Split the dataset on the best feature and recursively build subtrees
        feature_values = np.unique(X[:, best_feature])
        for value in feature_values:
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            # Remove the best feature column from the dataset
            sub_X = np.delete(sub_X, best_feature, axis=1)
            subtree = self._build_tree(sub_X, sub_y)
            tree[best_feature][value] = subtree
            
        return tree
    
    def _choose_best_feature_to_split(self, X, y):
        # Calculate the base entropy of the dataset
        base_entropy = self._calculate_entropy(y)
        best_info_gain = 0
        best_feature = None
        
        # Iterate over all features and calculate the information gain for each split
        for i in range(X.shape[1]):
            # Calculate the entropy of the dataset after splitting on this feature
            feature_values = np.unique(X[:, i])
            new_entropy = 0
            for value in feature_values:
                sub_y = y[X[:, i] == value]
                prob = len(sub_y) / len(y)
                new_entropy += prob * self._calculate_entropy(sub_y)
            
            # Calculate the information gain from this split
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        
        return best_feature

    def _predict_single(self, instance):
        # Traverse the tree to predict the class label of a single instance
        node = self.tree
        while isinstance(node, dict):  # while not a leaf node
            split_feature = next(iter(node))  # Get the feature to split on
            feature_value = instance[split_feature]
            if feature_value in node[split_feature]:
                node = node[split_feature][feature_value]
            else:
                # If the feature value is not in the tree (it was not during training), we stop and cannot predict
                return None
        # We have reached a leaf node which holds the class prediction
        return node

    def __str__(self) -> str:
        # Return a string representation of the ID3 algorithm
        return f"ID3 Decision Tree with configuration: {self.config}"

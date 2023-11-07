import numpy as np
import csv
import os
from sklearn.decomposition import PCA


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata  # deal with ASCII code

        unicodedata.numeric(s)  # change the string to float
        return True
    except (TypeError, ValueError):
        pass
        return False

def normalization(dataset):
    for i in range(len(dataset)):
        max_value = np.max(dataset[i])  # get the biggest value
        min_value = np.min(dataset[i])  # get the smallest value
        dataset[i] = (dataset[i] - min_value) / (
            max_value - min_value
        )  # do the normalization
    return dataset

def standardization(dataset):
    for i in range(len(dataset)):
        # get the mean and standard deviation
        mean = np.mean(dataset[i])
        std = np.std(dataset[i])
        # do the standardization
        dataset[i] = (dataset[i] - mean) / std
    return dataset

def apply_pca_if_needed(dataset, n_components=10):
    if dataset.shape[1] > n_components:  # Check if the number of features is greater than 10
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(dataset)
        return reduced_data
    else:
        return dataset
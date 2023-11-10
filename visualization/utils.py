import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

supported_visuals = """
confusion matrix
plotscatter2d
plotscatter3d
decision boundary
"""

def confusion_matrix(predictions, labels, num_classes=2):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(predictions)):
        pred = predictions[i]
        true = labels[i]
        confusion_matrix[pred][true] += 1
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    
def plotscatter2d(predictions, labels):
    plt.scatter(predictions[:, 0], predictions[:, 1], c=labels)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.show()
    
def plotscatter3d(predictions, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(predictions[:, 0], predictions[:, 1], predictions, c=labels)
    ax.set_xlabel('dimension1')
    ax.set_ylabel('dimension2')
    ax.set_zlabel('Target Value')
    plt.show()

def decision_boundary(predictions, labels, w, b, X, y):
    point_a = np.zeros((int(X.shape[0] / 2), X.shape[1]))
    point_b = np.zeros((int(X.shape[0] / 2), X.shape[1]))
    j = 0
    k = 0
    for i in range(X.shape[0]):
        if y[i] == 1:
            point_a[j] = X[i]
            j += 1
        else:
            point_b[k] = X[i]
            k += 1
    x = np.linspace(4, 6, 5)
    y = -(w[0] * x + b) / w[1]
    print("w: ", w)
    print("b: ", b)
    plt.scatter(point_a[:,0], point_a[:,1], color='orange')
    plt.scatter(point_b[:,0], point_b[:,1], color='green')
    plt.plot(x, y, color='purple')
    plt.show()
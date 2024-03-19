import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to implement K-Nearest Neighbors algorithm
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        nearest_neighbors = np.argsort(distances)[:k]
        nearest_labels = [y_train[idx] for idx in nearest_neighbors]
        predicted_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(predicted_label)
    return np.array(predictions)

# Function to visualize the data and the test points
def plot_clusters(X_train, y_train, X_test, y_test, k=3):
    plt.figure(figsize=(10, 6))

    # Plot training points
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for cluster, color in enumerate(colors):
        plt.scatter(X_train[y_train == cluster, 0], X_train[y_train == cluster, 1], label=f'Cluster {cluster}', color=color)

    # Plot test points
    for idx, test_point in enumerate(X_test):
        predicted_cluster = y_test[idx]
        plt.scatter(test_point[0], test_point[1], color=colors[predicted_cluster], edgecolor='black', marker='o', s=150, label=f'Test point {idx+1}')

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Clustered Iris dataset with KNN-ed Test Points, k = ' + str(k))
    plt.legend()
    plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random points to classify
random_points = np.array([[5.0, 3.5], [6.0, 2.5], [7.0, 4.0]])

# Predict the labels of the random points using KNN
predictions = knn_predict(X_train[:, :2], y_train, random_points, k=3)

# Visualize the clusters and the random points
plot_clusters(X_train[:, :2], y_train, random_points, predictions)

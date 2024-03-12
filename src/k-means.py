import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
k = 3 # Number of clusters
max_iter = 100 # Number of iterations

def initialize_centroids(data, k):
    # Randomly select k data points as initial centroids
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    return centroids

def assign_to_clusters(data, centroids):
    # Assign each data point to the nearest centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(data, clusters, k):
    # Update centroids based on the mean of data points in each cluster
    centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans(data, k, max_iter):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def plot_clusters(data, clusters, centroids):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for i in range(k):
        plt.scatter(data[clusters == i, 0], data[clusters == i, 1], c=colors[i], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='black', label='Centroids')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('K-means Clustering of Iris Dataset')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Perform K-means clustering
    clusters, centroids = kmeans(data[:, :2], k, max_iter)
    # Plot clusters and centroids
    plot_clusters(data[:, :2], clusters, centroids)

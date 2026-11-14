import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from time import time

def kmeans(X, k):
    """Cluster a numeric array into k groups and return centroids and labels."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels

diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")

def kmeans_diamonds(n, k):
    """Cluster the first n diamonds into k groups."""
    X = diamonds_numeric.head(n).to_numpy()

    centroids, labels = kmeans(X, k)

    return centroids, labels

def kmeans_timer(n, k, n_iter=5):
    """Return the average clustering runtime in seconds."""
    runtimes = []

    for _ in range(n_iter):
        start = time()

        kmeans_diamonds(n, k)

        elapsed = time() - start
        runtimes.append(elapsed)

    return sum(runtimes) / n_iter
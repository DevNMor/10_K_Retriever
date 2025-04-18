"""
Scaling, K selection, KMeans, and outlier detection.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def standard_scale_embeddings(X):
    """Standardize features to zero mean and unit variance."""
    return StandardScaler().fit_transform(X)


def choose_k_by_silhouette(X, k_min=2, k_max=10):
    """
    Evaluate silhouette score for k in [k_min..k_max], return best k.
    """
    best_k, best_s = k_min, -1
    for k in range(k_min, k_max+1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_s:
            best_s, best_k = s, k
    return best_k, None


def perform_kmeans(X, k):
    """
    Run KMeans, return labels and centroids.
    """
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    return km.labels_, km.cluster_centers_


def detect_outliers(X, labels, centroids, percentile=90):
    """
    Flag points whose distance to their centroid exceeds the given percentile.
    """
    d = np.linalg.norm(X - centroids[labels], axis=1)
    thr = np.percentile(d, percentile)
    return d > thr
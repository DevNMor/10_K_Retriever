"""
PCA and normalization utilities.
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def compute_pca(X, n_components, random_state=42):
    """
    Fit PCA to X, reduce to n_components dimensions.
    Returns transformed data and PCA model.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def normalize_rows(X):
    """
    L2-normalize each row vector to unit length (useful for cosine KMeans).
    """
    return normalize(X, norm='l2', axis=1)
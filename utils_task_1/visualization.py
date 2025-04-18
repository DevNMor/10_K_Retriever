"""
Plotting: PCA variance, clusters, outliers, and section mappings.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_pca_variance(X, save_path=None):
    """
    Plot cumulative explained variance ratio of PCA on X.
    """
    from sklearn.decomposition import PCA
    pca_full = PCA().fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    comps = np.arange(1, len(cum)+1)
    plt.figure(figsize=(8,5))
    plt.plot(comps, cum, marker='o')
    plt.xlabel('# Components')
    plt.ylabel('Cumulative Variance')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_clusters(e2d, labels, path=None):
    """Scatter 2D embeddings colored by cluster labels."""
    plt.figure(figsize=(8,6))
    sc = plt.scatter(e2d[:,0], e2d[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.title('t-SNE: colored by cluster')
    plt.colorbar(sc, label='Cluster')
    if path: plt.savefig(path)
    plt.show()


def plot_outliers(e2d, outliers, path=None):
    """Scatter 2D embeddings colored by outlier flags."""
    cols = np.where(outliers, 'red', 'blue')
    plt.figure(figsize=(8,6))
    plt.scatter(e2d[:,0], e2d[:,1], c=cols, alpha=0.7)
    plt.title('t-SNE: red = outlier')
    if path: plt.savefig(path)
    plt.show()


def plot_sections(e2d, section_labels, path=None, section_names=None):
    """Scatter 2D embeddings colored by section codes with legend."""
    section_to_code = {sec: i for i, sec in enumerate(section_names)}
    codes = np.array([section_to_code[s] for s in section_labels])
    plt.figure(figsize=(8,6))
    sc = plt.scatter(e2d[:,0], e2d[:,1], c=codes, cmap='tab20', alpha=0.7)
    plt.title('t-SNE: colored by section')
    if section_names:
        cb = plt.colorbar(sc, ticks=range(len(section_names)))
        cb.set_ticklabels(section_names)
    if path: plt.savefig(path)
    plt.show()
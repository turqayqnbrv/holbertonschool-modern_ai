#!/usr/bin/env python3
"""Module for performing PCA on tabular data."""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
    Performs PCA on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int|float|None): Number of components to keep,
            fraction of variance to preserve, or None to keep all.
        random_state (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Data transformed into principal component space.
        sklearn.decomposition.PCA: Fitted PCA instance.
    """
    pca = decomposition.PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance
        cov = np.cov(X.T)

    def transform(self, X):
        pass

from collections import Counter

import numpy as np

from Classifiers.DecisionTree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_trees=10,
        max_depth=10,
        min_samples_split=2,
        n_features=None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                self.max_depth,
                self.min_samples_split,
                n_features=self.n_features,
            )
            X_subset, y_subset = self._bootstrap(X, y)
            tree.fit(X_subset, y_subset)
            self.trees.append(tree)

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        final_preds = np.array([self._most_common_label(pred) for pred in tree_preds])
        return final_preds

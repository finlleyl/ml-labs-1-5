"""Random forests built from the local from-scratch CART implementation."""
from __future__ import annotations

import numpy as np
from .tree import DecisionTreeClassifierFromScratch, DecisionTreeRegressorFromScratch


class _RandomForestBase:
    _tree_class = None

    def __init__(self, n_estimators=50, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=22):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        seeds = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            seed = int(seeds.integers(0, np.iinfo(np.int32).max))
            bootstrap = np.random.default_rng(seed).integers(0, len(X), size=len(X))
            tree = self._tree_class(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X[bootstrap], y[bootstrap])
            self.estimators_.append(tree)
        return self


class RandomForestClassifierFromScratch(_RandomForestBase):
    _tree_class = DecisionTreeClassifierFromScratch

    def predict_proba(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before predict_proba")
        return np.mean([tree.predict_proba(X) for tree in self.estimators_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RandomForestRegressorFromScratch(_RandomForestBase):
    _tree_class = DecisionTreeRegressorFromScratch

    def __init__(self, n_estimators=50, max_depth=None, min_samples_leaf=1, max_features=1.0, random_state=22):
        super().__init__(n_estimators, max_depth, min_samples_leaf, max_features, random_state)

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before predict")
        return np.mean([tree.predict(X) for tree in self.estimators_], axis=0)

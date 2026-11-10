"""CART-style decision trees for binary classification and regression."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class _Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class _DecisionTreeBase:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, min_impurity_decrease=1e-12, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        if self._X.ndim != 2 or len(self._X) != len(self._y):
            raise ValueError("X must be 2-D and have the same length as y")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = self._X.shape[1]
        self.root_ = self._grow(self._X, self._y, 0)
        del self._X, self._y
        return self

    def _feature_subset(self, n_features):
        setting = self.max_features
        if setting is None:
            count = n_features
        elif setting == "sqrt":
            count = max(1, int(np.sqrt(n_features)))
        elif isinstance(setting, float):
            count = max(1, int(setting * n_features))
        else:
            count = min(n_features, int(setting))
        return self._rng.choice(n_features, size=count, replace=False)

    def _grow(self, X, y, depth):
        node = _Node(value=self._leaf_value(y))
        if len(y) < 2 * self.min_samples_leaf or self._is_pure(y):
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        parent_impurity = self._impurity(y)
        best_impurity = np.inf
        best_split = None
        for feature in self._feature_subset(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            sorted_x = X[order, feature]
            sorted_y = y[order]
            positions = np.flatnonzero(sorted_x[:-1] < sorted_x[1:]) + 1
            positions = positions[(positions >= self.min_samples_leaf) & (positions <= len(y) - self.min_samples_leaf)]
            if not len(positions):
                continue
            impurities = self._split_impurities(sorted_y, positions)
            local = int(np.argmin(impurities))
            if impurities[local] < best_impurity:
                split = int(positions[local])
                best_impurity = float(impurities[local])
                best_split = (feature, (sorted_x[split - 1] + sorted_x[split]) / 2.0, order, split)
        if best_split is None or parent_impurity - best_impurity <= self.min_impurity_decrease:
            return node
        feature, threshold, order, split = best_split
        left_indices, right_indices = order[:split], order[split:]
        node.feature = int(feature)
        node.threshold = float(threshold)
        node.left = self._grow(X[left_indices], y[left_indices], depth + 1)
        node.right = self._grow(X[right_indices], y[right_indices], depth + 1)
        return node

    def _predict_row(self, row):
        node = self.root_
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def _predict_values(self, X):
        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=np.float64)
        return np.asarray([self._predict_row(row) for row in X])


class DecisionTreeClassifierFromScratch(_DecisionTreeBase):
    def fit(self, X, y):
        labels = np.unique(y)
        if not set(labels).issubset({0, 1}):
            raise ValueError("This implementation expects binary labels 0/1")
        super().fit(X, y)
        self.classes_ = np.asarray([0, 1])
        return self

    def _is_pure(self, y):
        return np.all(y == y[0])

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _impurity(self, y):
        probability = np.mean(y)
        return 2.0 * probability * (1.0 - probability)

    def _split_impurities(self, y, positions):
        cumulative = np.concatenate([[0.0], np.cumsum(y)])
        n = len(y)
        left_n = positions.astype(float)
        right_n = n - left_n
        left_positive = cumulative[positions]
        right_positive = cumulative[-1] - left_positive
        left_p = left_positive / left_n
        right_p = right_positive / right_n
        return (left_n * 2 * left_p * (1 - left_p) + right_n * 2 * right_p * (1 - right_p)) / n

    def predict_proba(self, X):
        positive = np.clip(self._predict_values(X), 0.0, 1.0)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DecisionTreeRegressorFromScratch(_DecisionTreeBase):
    def _is_pure(self, y):
        return np.var(y) <= np.finfo(np.float64).eps

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _impurity(self, y):
        return float(np.var(y))

    def _split_impurities(self, y, positions):
        cumulative = np.concatenate([[0.0], np.cumsum(y)])
        cumulative_sq = np.concatenate([[0.0], np.cumsum(y * y)])
        n = len(y)
        left_n = positions.astype(float)
        right_n = n - left_n
        left_sum = cumulative[positions]
        right_sum = cumulative[-1] - left_sum
        left_sse = cumulative_sq[positions] - left_sum * left_sum / left_n
        right_sse = (cumulative_sq[-1] - cumulative_sq[positions]) - right_sum * right_sum / right_n
        return (left_sse + right_sse) / n

    def predict(self, X):
        return self._predict_values(X)

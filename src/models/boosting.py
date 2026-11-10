"""Gradient boosting with regression trees fitted to negative gradients."""
from __future__ import annotations

import numpy as np
from .tree import DecisionTreeRegressorFromScratch
from .linear import _sigmoid


class _GradientBoostingBase:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=3, random_state=22):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state

    def _fit_residuals(self, X, y, initial, raw_predictions):
        self.init_ = float(initial)
        self.estimators_ = []
        current = np.full(len(y), self.init_, dtype=np.float64) if raw_predictions is None else raw_predictions.copy()
        for _ in range(self.n_estimators):
            residual = self._negative_gradient(y, current)
            tree = DecisionTreeRegressorFromScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree.fit(X, residual)
            current += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def _raw_predict(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before predict")
        result = np.full(len(X), self.init_, dtype=np.float64)
        for tree in self.estimators_:
            result += self.learning_rate * tree.predict(X)
        return result


class GradientBoostingClassifierFromScratch(_GradientBoostingBase):
    """Binary logistic-loss gradient boosting."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("This implementation expects binary labels 0/1")
        probability = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return self._fit_residuals(X, y, np.log(probability / (1 - probability)), None)

    def _negative_gradient(self, y, raw_predictions):
        return y - _sigmoid(raw_predictions)

    def predict_proba(self, X):
        positive = _sigmoid(self._raw_predict(X))
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GradientBoostingRegressorFromScratch(_GradientBoostingBase):
    """Squared-error gradient boosting; residuals are y - F(x)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return self._fit_residuals(X, y, np.mean(y), None)

    def _negative_gradient(self, y, raw_predictions):
        return y - raw_predictions

    def predict(self, X):
        return self._raw_predict(X)

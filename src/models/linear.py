"""Linear and logistic regression implemented with NumPy."""
from __future__ import annotations

import numpy as np


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


class LogisticRegressionFromScratch:
    """Binary logistic regression trained by full-batch gradient descent."""

    def __init__(self, learning_rate=0.1, max_iter=5000, l2=0.0, class_weight=None, tol=1e-8):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.tol = float(tol)

    def _sample_weights(self, y):
        if self.class_weight is None:
            return np.ones(len(y), dtype=np.float64)
        if self.class_weight == "balanced":
            counts = np.bincount(y.astype(int), minlength=2)
            return np.where(y == 1, len(y) / (2 * counts[1]), len(y) / (2 * counts[0]))
        if isinstance(self.class_weight, dict):
            return np.asarray([self.class_weight.get(int(label), 1.0) for label in y])
        raise ValueError("class_weight must be None, 'balanced' or a dictionary")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y) or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Expected 2-D X and binary y (0/1)")
        design = np.column_stack([np.ones(len(X)), X])
        sample_weights = self._sample_weights(y)
        normalizer = sample_weights.sum()
        weights = np.zeros(design.shape[1], dtype=np.float64)
        previous_loss = np.inf
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            linear = design @ weights
            probabilities = _sigmoid(linear)
            gradient = design.T @ ((probabilities - y) * sample_weights) / normalizer
            gradient[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * gradient
            self.n_iter_ = iteration
            if iteration % 25 == 0:
                probabilities = np.clip(_sigmoid(design @ weights), 1e-12, 1 - 1e-12)
                loss = -np.sum(sample_weights * (y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))) / normalizer
                loss += 0.5 * self.l2 * np.dot(weights[1:], weights[1:])
                if abs(previous_loss - loss) < self.tol:
                    break
                previous_loss = loss
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self

    def predict_proba(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict_proba")
        positive = _sigmoid(self.intercept_ + np.asarray(X, dtype=np.float64) @ self.coef_)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LinearRegressionFromScratch:
    """Ordinary/ridge least squares with an unpenalized intercept."""

    def __init__(self, l2=0.0):
        self.l2 = float(l2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        design = np.column_stack([np.ones(len(X)), X])
        if self.l2 > 0:
            penalty = np.eye(design.shape[1]) * self.l2
            penalty[0, 0] = 0.0
            weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        else:
            # lstsq is stable for collinearity, unlike explicit matrix inversion.
            weights = np.linalg.lstsq(design, y, rcond=None)[0]
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict")
        return self.intercept_ + np.asarray(X, dtype=np.float64) @ self.coef_

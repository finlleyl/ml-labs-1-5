"""K-nearest neighbours implemented with NumPy only."""
from __future__ import annotations

import numpy as np


class _KNNBase:
    def __init__(self, n_neighbors=5, weights="uniform", p=2, chunk_size=64):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = p
        self.chunk_size = int(chunk_size)

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y)
        if self._X.ndim != 2 or len(self._X) != len(self._y):
            raise ValueError("X must be 2-D and have the same length as y")
        if not 1 <= self.n_neighbors <= len(self._X):
            raise ValueError("n_neighbors must be in [1, n_samples]")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if self.p < 1:
            raise ValueError("p must be >= 1")
        if not np.isfinite(self._X).all():
            raise ValueError("X contains NaN or infinity")
        return self

    def _distance(self, X):
        if self.p == 2:
            # Chunking bounds memory: a full test x train x features tensor is not created.
            result = (
                np.sum(X * X, axis=1, keepdims=True)
                + np.sum(self._X * self._X, axis=1)[None, :]
                - 2.0 * X @ self._X.T
            )
            return np.sqrt(np.maximum(result, 0.0))
        if self.p == 1:
            return np.sum(np.abs(X[:, None, :] - self._X[None, :, :]), axis=2)
        return np.sum(np.abs(X[:, None, :] - self._X[None, :, :]) ** self.p, axis=2) ** (1.0 / self.p)

    def _neighbors(self, X):
        X = np.asarray(X, dtype=np.float64)
        for start in range(0, len(X), self.chunk_size):
            distances = self._distance(X[start : start + self.chunk_size])
            indices = np.argpartition(distances, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
            selected = np.take_along_axis(distances, indices, axis=1)
            order = np.argsort(selected, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
            selected = np.take_along_axis(selected, order, axis=1)
            for row_indices, row_distances in zip(indices, selected):
                yield self._y[row_indices], row_distances

    def _weights(self, distances):
        if self.weights == "uniform":
            return np.ones_like(distances)
        # Zero-distance neighbours dominate instead of causing a division by zero.
        if np.any(distances == 0):
            return (distances == 0).astype(np.float64)
        return 1.0 / np.maximum(distances, np.finfo(np.float64).eps)


class KNNClassifierFromScratch(_KNNBase):
    """Binary/multiclass KNN; predict_proba returns weighted neighbour frequencies."""

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_, self._encoded_y = np.unique(self._y, return_inverse=True)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "classes_"):
            raise RuntimeError("Call fit before predict_proba")
        probabilities = []
        for labels, distances in self._neighbors(X):
            weights = self._weights(distances)
            encoded = np.searchsorted(self.classes_, labels)
            probabilities.append([weights[encoded == i].sum() for i in range(len(self.classes_))])
        probabilities = np.asarray(probabilities, dtype=np.float64)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNNRegressorFromScratch(_KNNBase):
    def fit(self, X, y):
        super().fit(X, y)
        self._y = self._y.astype(np.float64)
        return self

    def predict(self, X):
        if not hasattr(self, "_X"):
            raise RuntimeError("Call fit before predict")
        predictions = []
        for labels, distances in self._neighbors(X):
            weights = self._weights(distances)
            predictions.append(np.dot(labels, weights) / weights.sum())
        return np.asarray(predictions)

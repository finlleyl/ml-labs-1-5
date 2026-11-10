"""Shared experiment pipeline used by all five laboratory notebooks."""
from __future__ import annotations

import json
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .data import RANDOM_STATE, load_dataset
from .models import (
    DecisionTreeClassifierFromScratch, DecisionTreeRegressorFromScratch,
    GradientBoostingClassifierFromScratch, GradientBoostingRegressorFromScratch,
    KNNClassifierFromScratch, KNNRegressorFromScratch,
    LinearRegressionFromScratch, LogisticRegressionFromScratch,
    RandomForestClassifierFromScratch, RandomForestRegressorFromScratch,
)

LAB_NAMES = {
    1: "KNN",
    2: "Логистическая / линейная регрессия",
    3: "Решающее дерево",
    4: "Случайный лес",
    5: "Градиентный бустинг",
}


def _json_params(params):
    return json.dumps(params, ensure_ascii=False, default=str, sort_keys=True)


def _sklearn_model(lab, task, improved=False):
    if lab == 1:
        return KNeighborsClassifier() if task == "classification" else KNeighborsRegressor()
    if lab == 2:
        if task == "classification":
            return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        return LinearRegression()
    if lab == 3:
        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        return cls(random_state=RANDOM_STATE)
    if lab == 4:
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        kwargs = {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1}
        if improved:
            kwargs["max_features"] = "sqrt" if task == "classification" else 0.8
        return cls(**kwargs)
    if lab == 5:
        cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        return cls(random_state=RANDOM_STATE)
    raise ValueError("lab must be in 1..5")


def _pipeline(lab, task, improved=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if improved:
        steps.append(("variance", VarianceThreshold(0.0)))
        if lab == 2 and task == "regression":
            steps.append(("poly", PolynomialFeatures(include_bias=False)))
        if lab in {1, 2}:
            steps.append(("scaler", StandardScaler()))
    steps.append(("model", _sklearn_model(lab, task, improved)))
    return Pipeline(steps)


def _param_grid(lab, task):
    if lab == 1:
        return {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"], "model__p": [1, 2]}
    if lab == 2 and task == "classification":
        return {"model__C": [0.05, 0.2, 1.0, 5.0], "model__class_weight": [None, "balanced"]}
    if lab == 2:
        return {"poly__degree": [1, 2]}
    if lab == 3:
        return {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 7]}
    if lab == 4:
        return {"model__n_estimators": [80, 150], "model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3]}
    if lab == 5:
        return {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]}
    raise ValueError("Unknown experiment")


def _custom_baseline(lab, task):
    if lab == 1:
        return KNNClassifierFromScratch() if task == "classification" else KNNRegressorFromScratch()
    if lab == 2:
        return LogisticRegressionFromScratch() if task == "classification" else LinearRegressionFromScratch()
    if lab == 3:
        cls = DecisionTreeClassifierFromScratch if task == "classification" else DecisionTreeRegressorFromScratch
        return cls(random_state=RANDOM_STATE)
    if lab == 4:
        if task == "classification":
            return RandomForestClassifierFromScratch(n_estimators=30, random_state=RANDOM_STATE)
        return RandomForestRegressorFromScratch(n_estimators=30, random_state=RANDOM_STATE)
    if lab == 5:
        cls = GradientBoostingClassifierFromScratch if task == "classification" else GradientBoostingRegressorFromScratch
        return cls(random_state=RANDOM_STATE)
    raise ValueError("lab must be in 1..5")


def _custom_improved(lab, task, best_params, n_samples, max_features):
    params = {key.removeprefix("model__"): value for key, value in best_params.items() if key.startswith("model__")}
    if lab == 1:
        cls = KNNClassifierFromScratch if task == "classification" else KNNRegressorFromScratch
        return cls(**params)
    if lab == 2 and task == "classification":
        # sklearn minimizes sum(loss) + ||w||^2/(2C); ours uses mean(loss) + l2||w||^2/2.
        params["l2"] = 1.0 / (params.pop("C") * n_samples)
        return LogisticRegressionFromScratch(**params)
    if lab == 2:
        return LinearRegressionFromScratch()
    if lab == 3:
        cls = DecisionTreeClassifierFromScratch if task == "classification" else DecisionTreeRegressorFromScratch
        return cls(random_state=RANDOM_STATE, **params)
    if lab == 4:
        cls = RandomForestClassifierFromScratch if task == "classification" else RandomForestRegressorFromScratch
        # Fifty trees keep the pure-Python experiment practical; all other selected parameters are retained.
        params["n_estimators"] = min(params["n_estimators"], 50)
        return cls(max_features=max_features, random_state=RANDOM_STATE, **params)
    if lab == 5:
        cls = GradientBoostingClassifierFromScratch if task == "classification" else GradientBoostingRegressorFromScratch
        return cls(random_state=RANDOM_STATE, **params)
    raise ValueError("lab must be in 1..5")


class TaskExperiment:
    """Stateful experiment: each method adds a comparable test-set result."""

    def __init__(self, lab, task):
        self.lab = int(lab)
        self.task = task
        self.algorithm = LAB_NAMES[self.lab]
        self.data = load_dataset(task)
        self.rows = []
        self.baseline_pipeline = _pipeline(self.lab, self.task, improved=False)
        self.improved_pipeline = _pipeline(self.lab, self.task, improved=True)

    def _metrics(self, model, X, y):
        prediction = model.predict(X)
        if self.task == "classification":
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X)[:, 1]
            else:
                score = model.decision_function(X)
            return {"accuracy": accuracy_score(y, prediction), "f1": f1_score(y, prediction), "roc_auc": roc_auc_score(y, score)}
        return {"rmse": root_mean_squared_error(y, prediction), "mae": mean_absolute_error(y, prediction), "r2": r2_score(y, prediction)}

    def _record(self, stage, implementation, model, X, seconds, params=None):
        metric_values = self._metrics(model, X, self.data.y_test)
        row = {
            "lab": self.lab, "algorithm": self.algorithm, "task": self.task,
            "implementation": implementation, "stage": stage,
            **metric_values, "fit_seconds": seconds, "params": _json_params(params or {}),
        }
        self.rows.append(row)
        metrics = ", ".join(f"{key}={value:.4f}" for key, value in metric_values.items())
        print(f"{self.task} | {implementation} | {stage}: {metrics}; {seconds:.2f} s")
        return row

    def fit_sklearn_baseline(self):
        start = perf_counter()
        self.baseline_pipeline.fit(self.data.X_train, self.data.y_train)
        seconds = perf_counter() - start
        self._record("baseline", "sklearn", self.baseline_pipeline, self.data.X_test, seconds)
        return self

    def fit_sklearn_improved(self):
        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE) if self.task == "classification" else KFold(5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "roc_auc" if self.task == "classification" else "neg_root_mean_squared_error"
        self.search = GridSearchCV(
            self.improved_pipeline, _param_grid(self.lab, self.task), scoring=scoring,
            cv=cv, n_jobs=-1, refit=True, return_train_score=False,
        )
        start = perf_counter()
        self.search.fit(self.data.X_train, self.data.y_train)
        seconds = perf_counter() - start
        self.best_model = self.search.best_estimator_
        self._record("improved", "sklearn", self.best_model, self.data.X_test, seconds, self.search.best_params_)
        return self

    def fit_custom_baseline(self):
        preprocessor = self.baseline_pipeline.named_steps["imputer"]
        X_train = preprocessor.transform(self.data.X_train)
        X_test = preprocessor.transform(self.data.X_test)
        self.custom_baseline = _custom_baseline(self.lab, self.task)
        start = perf_counter()
        self.custom_baseline.fit(X_train, self.data.y_train.to_numpy())
        seconds = perf_counter() - start
        self._record("baseline", "from_scratch", self.custom_baseline, X_test, seconds)
        return self

    def fit_custom_improved(self):
        # Slicing retains every fitted preprocessing step and excludes the sklearn estimator.
        preprocessor = Pipeline(self.best_model.steps[:-1])
        X_train = preprocessor.transform(self.data.X_train)
        X_test = preprocessor.transform(self.data.X_test)
        max_features = self.best_model.named_steps["model"].max_features if self.lab == 4 else None
        self.custom_improved = _custom_improved(self.lab, self.task, self.search.best_params_, len(self.data.y_train), max_features)
        start = perf_counter()
        self.custom_improved.fit(X_train, self.data.y_train.to_numpy())
        seconds = perf_counter() - start
        reported_params = dict(self.search.best_params_)
        if self.lab == 2 and self.task == "classification":
            reported_params["custom__l2"] = self.custom_improved.l2
        if self.lab == 4:
            reported_params["custom__n_estimators"] = self.custom_improved.n_estimators
        self._record("improved", "from_scratch", self.custom_improved, X_test, seconds, reported_params)
        return self

    @property
    def results(self):
        columns = ["lab", "algorithm", "task", "implementation", "stage", "accuracy", "f1", "roc_auc", "rmse", "mae", "r2", "fit_seconds", "params"]
        return pd.DataFrame(self.rows).reindex(columns=columns)

    def cv_table(self):
        table = pd.DataFrame(self.search.cv_results_)
        result = table[["params", "mean_test_score", "std_test_score", "mean_fit_time"]].copy()
        if self.task == "regression":
            result["mean_test_score"] = -result["mean_test_score"]
            result = result.rename(columns={"mean_test_score": "cv_rmse", "std_test_score": "std_rmse"})
        else:
            result = result.rename(columns={"mean_test_score": "cv_roc_auc", "std_test_score": "std_roc_auc"})
        return result.sort_values(result.columns[1], ascending=self.task == "regression").reset_index(drop=True)


def run_all_experiments():
    """CLI helper; notebooks deliberately invoke each stage separately for readability."""
    frames = []
    for lab in range(1, 6):
        for task in ("classification", "regression"):
            experiment = TaskExperiment(lab, task)
            experiment.fit_sklearn_baseline().fit_sklearn_improved().fit_custom_baseline().fit_custom_improved()
            frames.append(experiment.results)
    return pd.concat(frames, ignore_index=True)

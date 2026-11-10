"""Reproducible loading and fixed train/test splits for both UCI datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 22


@dataclass
class DatasetSplit:
    task: str
    name: str
    practical_task: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]

    @property
    def X(self):
        return pd.concat([self.X_train, self.X_test]).sort_index()

    @property
    def y(self):
        return pd.concat([self.y_train, self.y_test]).sort_index()


def load_classification() -> DatasetSplit:
    path = ROOT / "data" / "ionosphere.data"
    columns = [f"signal_{i:02d}" for i in range(1, 35)] + ["target"]
    frame = pd.read_csv(path, header=None, names=columns)
    # bad=1 is the anomaly/negative-structure class used as the positive class in F1.
    frame["target"] = frame["target"].map({"g": 0, "b": 1}).astype(int)
    X = frame.drop(columns="target")
    y = frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return DatasetSplit(
        task="classification",
        name="Ionosphere (UCI)",
        practical_task="По сигналам фазированной антенной решётки отличать радиосигналы со структурой в ионосфере от проходящих сигналов.",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )


def load_regression() -> DatasetSplit:
    path = ROOT / "data" / "Concrete_Data.xls"
    frame = pd.read_excel(path)
    frame.columns = [
        "cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer",
        "coarse_aggregate", "fine_aggregate", "age_days", "strength_mpa",
    ]
    X = frame.drop(columns="strength_mpa")
    y = frame["strength_mpa"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return DatasetSplit(
        task="regression",
        name="Concrete Compressive Strength (UCI)",
        practical_task="По составу смеси и возрасту бетона прогнозировать прочность на сжатие в МПа для предварительной оценки рецептуры.",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )


def load_dataset(task: str) -> DatasetSplit:
    if task == "classification":
        return load_classification()
    if task == "regression":
        return load_regression()
    raise ValueError("task must be 'classification' or 'regression'")

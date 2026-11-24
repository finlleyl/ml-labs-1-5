"""Загрузка наборов UCI и одинаковые разбиения для всех работ."""
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
    path = ROOT / "data" / "haberman.data"
    columns = ["age", "operation_year", "positive_axillary_nodes", "survival_status"]
    frame = pd.read_csv(path, header=None, names=columns)
    # Класс 1 означает смерть в течение пяти лет после операции.
    frame["survival_status"] = frame["survival_status"].map({1: 0, 2: 1}).astype(int)
    X = frame.drop(columns="survival_status")
    y = frame["survival_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return DatasetSplit(
        task="classification",
        name="Haberman's Survival (UCI)",
        practical_task="По возрасту пациента, году операции и числу пораженных лимфоузлов оценить риск смерти в течение пяти лет после операции.",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )


def load_regression() -> DatasetSplit:
    path = ROOT / "data" / "qsar_aquatic_toxicity.csv"
    columns = ["tpsa_tot", "saacc", "h_050", "mlogp", "rdchi", "gats1p", "n_n", "c_040", "lc50"]
    frame = pd.read_csv(path, sep=";", header=None, names=columns)
    X = frame.drop(columns="lc50")
    y = frame["lc50"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return DatasetSplit(
        task="regression",
        name="QSAR aquatic toxicity (UCI)",
        practical_task="По молекулярным дескрипторам прогнозировать токсичность LC50 для Daphnia magna и предварительно оценивать опасность веществ для водных организмов.",
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

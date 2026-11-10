import unittest
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from src.data import load_classification, load_regression
from src.models import (
    DecisionTreeClassifierFromScratch, DecisionTreeRegressorFromScratch,
    GradientBoostingClassifierFromScratch, GradientBoostingRegressorFromScratch,
    KNNClassifierFromScratch, KNNRegressorFromScratch,
    LinearRegressionFromScratch, LogisticRegressionFromScratch,
    RandomForestClassifierFromScratch, RandomForestRegressorFromScratch,
)


class DataTests(unittest.TestCase):
    def test_real_datasets_and_fixed_splits(self):
        classification = load_classification()
        regression = load_regression()
        self.assertEqual((len(classification.X), len(classification.X_test)), (351, 71))
        self.assertEqual((len(regression.X), len(regression.X_test)), (1030, 206))
        self.assertEqual(set(classification.y.unique()), {0, 1})
        self.assertFalse(classification.X.isna().any().any())
        self.assertFalse(regression.X.isna().any().any())
        self.assertEqual(classification.X_train.index.tolist(), load_classification().X_train.index.tolist())
        self.assertEqual(regression.X_train.index.tolist(), load_regression().X_train.index.tolist())


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(22)
        cls.Xc = rng.normal(size=(100, 4))
        logits = 2.2 * cls.Xc[:, 0] - 1.5 * cls.Xc[:, 1] + 0.6 * cls.Xc[:, 2]
        cls.yc = (logits + rng.normal(scale=0.4, size=100) > 0).astype(int)
        cls.Xr = rng.normal(size=(100, 4))
        cls.yr = 3 * cls.Xr[:, 0] - 2 * cls.Xr[:, 1] + 0.4 * cls.Xr[:, 2] ** 2 + rng.normal(scale=0.2, size=100)

    def test_classifiers(self):
        models = [
            KNNClassifierFromScratch(n_neighbors=5),
            LogisticRegressionFromScratch(max_iter=2500),
            DecisionTreeClassifierFromScratch(max_depth=5, random_state=22),
            RandomForestClassifierFromScratch(n_estimators=5, max_depth=5, random_state=22),
            GradientBoostingClassifierFromScratch(n_estimators=15, max_depth=2, random_state=22),
        ]
        scaler = StandardScaler().fit(self.Xc[:80])
        for model in models:
            with self.subTest(model=type(model).__name__):
                X = scaler.transform(self.Xc) if isinstance(model, (KNNClassifierFromScratch, LogisticRegressionFromScratch)) else self.Xc
                model.fit(X[:80], self.yc[:80])
                probabilities = model.predict_proba(X[80:])
                self.assertEqual(probabilities.shape, (20, 2))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
                self.assertGreaterEqual(accuracy_score(self.yc[80:], model.predict(X[80:])), 0.75)

    def test_regressors(self):
        models = [
            KNNRegressorFromScratch(n_neighbors=5),
            LinearRegressionFromScratch(),
            DecisionTreeRegressorFromScratch(max_depth=5, random_state=22),
            RandomForestRegressorFromScratch(n_estimators=5, max_depth=5, random_state=22),
            GradientBoostingRegressorFromScratch(n_estimators=15, max_depth=2, random_state=22),
        ]
        scaler = StandardScaler().fit(self.Xr[:80])
        for model in models:
            with self.subTest(model=type(model).__name__):
                X = scaler.transform(self.Xr) if isinstance(model, KNNRegressorFromScratch) else self.Xr
                model.fit(X[:80], self.yr[:80])
                predictions = model.predict(X[80:])
                self.assertEqual(predictions.shape, (20,))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertGreaterEqual(r2_score(self.yr[80:], predictions), 0.35)


if __name__ == "__main__":
    unittest.main()

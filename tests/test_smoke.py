import unittest
import numpy as np

from agriformer.metrics import compute_metrics
from agriformer.preparation import FoldScope, PLSContextTransform
from agriformer.splits import fixed_holdout


class SmokeTests(unittest.TestCase):
    def test_synthetic_pls_pipeline(self):
        rng = np.random.default_rng(20261028)
        X = rng.normal(size=(40, 20))
        X[0, 0] = np.nan
        y = np.nan_to_num(X[:, :3], nan=0.0).sum(axis=1) + rng.normal(0, 0.1, 40)
        train, test = fixed_holdout(len(y), seed=42)
        fitted = PLSContextTransform.fit(X, y, FoldScope(train, test), n_components=3)
        train_scores = fitted.transform(X, train)
        test_scores = fitted.transform(X, test)
        coefficients, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(train)), train_scores]), y[train], rcond=None
        )
        prediction = np.column_stack([np.ones(len(test)), test_scores]) @ coefficients
        metrics = compute_metrics(y[test], prediction)
        self.assertEqual(metrics["n"], len(test))
        self.assertTrue(np.isfinite(metrics["pcc"]))


if __name__ == "__main__":
    unittest.main()

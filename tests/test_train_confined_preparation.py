import unittest
import numpy as np

from agriformer.preparation import (
    FoldScope,
    GenotypeTransform,
    MarkerMeanTransform,
    select_pls_dimension,
)


class PreparationTests(unittest.TestCase):
    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            FoldScope(np.array([0, 1]), np.array([1, 2]))

    def test_test_values_do_not_affect_training_means(self):
        X = np.array([[0.0, np.nan], [2.0, 4.0], [999.0, 999.0]])
        scope = FoldScope(np.array([0, 1]), np.array([2]))
        fitted = MarkerMeanTransform.fit(X, scope)
        np.testing.assert_allclose(fitted.means, [1.0, 4.0])

    def test_training_mean_genotype_mode_uses_training_rows_only(self):
        X = np.array([[0.0, -1.0], [2.0, 2.0], [999.0, 999.0]])
        scope = FoldScope(np.array([0, 1]), np.array([2]))
        fitted = GenotypeTransform.fit(X, scope, "training_mean")
        np.testing.assert_allclose(fitted.marker_means, [1.0, 2.0])
        np.testing.assert_allclose(fitted.transform_rows(np.array([[-1.0, -1.0]])), [[1.0, 2.0]])

    def test_original_genotype_mode_preserves_minus_one(self):
        X = np.array([[0.0, -1.0], [2.0, 2.0]])
        scope = FoldScope(np.array([0]), np.array([1]))
        fitted = GenotypeTransform.fit(X, scope, "original")
        np.testing.assert_allclose(fitted.transform(X, [0]), [[0.0, -1.0]])

    def test_pls_selection_does_not_read_outer_test_labels(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(15, 5))
        y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=15)
        scope = FoldScope(np.arange(12), np.arange(12, 15))
        selected_left, scores_left = select_pls_dimension(
            X, y, scope, "original", candidates=(1, 2), folds=3, seed=7
        )
        changed = y.copy()
        changed[scope.test_indices] = 1e12
        selected_right, scores_right = select_pls_dimension(
            X, changed, scope, "original", candidates=(1, 2), folds=3, seed=7
        )
        self.assertEqual(selected_left, selected_right)
        self.assertEqual(scores_left, scores_right)


if __name__ == "__main__":
    unittest.main()


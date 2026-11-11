import unittest
import numpy as np

from agriformer.metrics import compute_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 4.0, 8.0])
        result = compute_metrics(y, y)
        self.assertAlmostEqual(result["pcc"], 1.0)
        self.assertAlmostEqual(result["ccc_population"], 1.0)
        self.assertAlmostEqual(result["r2"], 1.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["slope_b1"], 1.0)

    def test_nonfinite_pairs_are_removed(self):
        result = compute_metrics([1, 2, np.nan, 4], [1, 2, 3, np.inf])
        self.assertEqual(result["n"], 2)


if __name__ == "__main__":
    unittest.main()


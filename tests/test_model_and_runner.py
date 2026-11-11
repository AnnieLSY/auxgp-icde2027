import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np
import pandas as pd
import torch

from agriformer.learning import fit_predict_agriformer
from agriformer.model import AgriFormer
from agriformer.runner import parser, run
from agriformer.baselines import fit_predict_r


class ModelAndRunnerTests(unittest.TestCase):
    def test_model_output_shape_and_complete_trait_assignment(self):
        model = AgriFormer(4, 6, 3, {"high": [0], "moderate": [1], "low": [2], "independent": []})
        prediction = model(torch.randn(5, 4), torch.randn(5, 6))
        self.assertEqual(tuple(prediction.shape), (5, 3))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_one_epoch_agriformer_smoke(self):
        rng = np.random.default_rng(4)
        context_train = rng.normal(size=(12, 3)).astype("float32")
        effects_train = rng.normal(size=(12, 8)).astype("float32")
        targets = np.column_stack([context_train[:, 0], effects_train[:, 0]]).astype("float32")
        prediction, _, metadata = fit_predict_agriformer(
            context_train, effects_train, targets,
            rng.normal(size=(4, 3)).astype("float32"), rng.normal(size=(4, 8)).astype("float32"),
            target_index=0, trait_names=["yield", "aux"], dataset="synthetic",
            seed=5, epochs=1, batch_size=4, device="cpu",
        )
        self.assertEqual(prediction.shape, (4, 2))
        self.assertEqual(metadata["epochs"], 1)

    def test_unified_ridge_entry_writes_common_schema(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); ids = [f"s{i}" for i in range(24)]
            X = rng.normal(size=(24, 8)); y = X[:, 0] + rng.normal(scale=.1, size=24)
            pd.DataFrame(X, columns=[f"m{i}" for i in range(8)]).assign(material_id=ids).to_csv(folder / "X.csv", index=False)
            pd.DataFrame({"material_id": ids, "yield": y}).to_csv(folder / "Y.csv", index=False)
            pd.DataFrame({"material_id": ids, "partition": ["train"] * 18 + ["test"] * 6}).to_csv(folder / "split.csv", index=False)
            args = parser().parse_args([
                "--model", "ridge", "--genotypes", str(folder / "X.csv"), "--phenotypes", str(folder / "Y.csv"),
                "--split-manifest", str(folder / "split.csv"), "--target-index", "0", "--dataset", "synthetic",
                "--out", str(folder / "out"),
            ])
            metrics = run(args)
            self.assertEqual(metrics["n"], 6)
            self.assertTrue((folder / "out/predictions.csv").exists())
            self.assertEqual(json.loads((folder / "out/run.json").read_text())["model"], "ridge")

    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is not installed")
    def test_unified_agriformer_entry_includes_preparation_identity(self):
        rng = np.random.default_rng(18)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); ids = [f"s{i}" for i in range(24)]
            X = rng.normal(size=(24, 8)); y = X[:, 0] - .3 * X[:, 3] + rng.normal(scale=.1, size=24)
            pd.DataFrame(X, columns=[f"m{i}" for i in range(8)]).assign(material_id=ids).to_csv(folder / "X.csv", index=False)
            pd.DataFrame({"material_id": ids, "yield": y}).to_csv(folder / "Y.csv", index=False)
            pd.DataFrame({"material_id": ids, "partition": ["train"] * 18 + ["test"] * 6}).to_csv(folder / "split.csv", index=False)
            args = parser().parse_args([
                "--model", "agriformer", "--genotypes", str(folder / "X.csv"), "--phenotypes", str(folder / "Y.csv"),
                "--split-manifest", str(folder / "split.csv"), "--target-index", "0", "--dataset", "synthetic",
                "--top-k", "5", "--epochs", "1", "--device", "cpu", "--out", str(folder / "out"),
            ])
            metrics = run(args)
            self.assertEqual(metrics["n"], 6)
            self.assertTrue((folder / "out/preparation/fitting_rows.csv").exists())
            self.assertTrue((folder / "out/preparation/selected_markers.csv").exists())
            self.assertTrue((folder / "out/preparation/pls_state.json").exists())

    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is not installed")
    def test_frozen_r_baselines_return_finite_predictions(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(28, 10)); y = X[:, 0] + rng.normal(scale=.2, size=28)
        for model in ("gblup", "bayesb"):
            prediction, _ = fit_predict_r(model, X[:22], y[:22], X[22:], seed=11)
            self.assertEqual(len(prediction), 6)
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()

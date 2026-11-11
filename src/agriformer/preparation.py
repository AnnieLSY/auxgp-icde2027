"""Train-confined preparation primitives with explicit leakage guards.

The public API requires an explicit :class:`FoldScope`.  Label-derived state is
fit from ``train_indices`` only; evaluation labels are never accepted by a fit
method.  Dataset-specific genotype handling is applied before PLS projection
and BayesB effect estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .splits import assert_disjoint


@dataclass(frozen=True)
class FoldScope:
    train_indices: np.ndarray
    test_indices: np.ndarray

    def __post_init__(self):
        assert_disjoint(self.train_indices, self.test_indices)


@dataclass
class MarkerMeanTransform:
    means: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, scope: FoldScope):
        train = np.asarray(scope.train_indices, dtype=int)
        means = np.nanmean(np.asarray(X, dtype=float)[train], axis=0)
        means[~np.isfinite(means)] = 0.0
        return cls(means=means)

    def transform(self, X: np.ndarray, indices):
        values = np.asarray(X, dtype=float)[np.asarray(indices, dtype=int)]
        return np.where(np.isnan(values), self.means, values)


@dataclass
class GenotypeTransform:
    """Dataset-specific genotype handling fitted on training rows only.

    ``training_mean`` treats both ``NaN`` and the numerical ``-1`` sentinel as
    missing.  ``original`` preserves every numerical value, including ``-1``.
    ``supplied_preprocessed`` is an explicit alias for an unchanged upstream
    matrix and records that no local imputation is performed.
    """

    mode: str
    marker_means: np.ndarray | None = None

    @classmethod
    def fit(cls, X: np.ndarray, scope: FoldScope, mode: str):
        allowed = {"training_mean", "original", "supplied_preprocessed"}
        if mode not in allowed:
            raise ValueError(f"unsupported genotype mode: {mode!r}")
        if mode != "training_mean":
            return cls(mode=mode, marker_means=None)
        train = np.asarray(X, dtype=float)[np.asarray(scope.train_indices, dtype=int)]
        observed = np.where((train == -1) | np.isnan(train), np.nan, train)
        with np.errstate(all="ignore"):
            means = np.nanmean(observed, axis=0)
        means[~np.isfinite(means)] = 1.0
        return cls(mode=mode, marker_means=means)

    def transform_rows(self, values: np.ndarray):
        output = np.asarray(values, dtype=float).copy()
        if self.mode == "training_mean":
            missing = (output == -1) | np.isnan(output)
            output[missing] = np.broadcast_to(self.marker_means, output.shape)[missing]
        return output

    def transform(self, X: np.ndarray, indices):
        rows = np.asarray(X, dtype=float)[np.asarray(indices, dtype=int)]
        return self.transform_rows(rows)


@dataclass
class PLSContextTransform:
    imputer: MarkerMeanTransform
    model: PLSRegression

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, scope: FoldScope, n_components: int):
        train = np.asarray(scope.train_indices, dtype=int)
        imputer = MarkerMeanTransform.fit(X, scope)
        X_train = imputer.transform(X, train)
        y_train = np.asarray(y, dtype=float)[train]
        model = PLSRegression(n_components=int(n_components), scale=True)
        model.fit(X_train, y_train)
        return cls(imputer=imputer, model=model)

    def transform(self, X: np.ndarray, indices):
        values = self.imputer.transform(X, indices)
        return self.model.transform(values)


@dataclass
class TrainScopedPLS:
    """PLS state whose standardization and fit are confined to a fold."""

    genotype: GenotypeTransform
    scaler: StandardScaler
    model: PLSRegression

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        scope: FoldScope,
        n_components: int,
        genotype_mode: str,
    ):
        train = np.asarray(scope.train_indices, dtype=int)
        genotype = GenotypeTransform.fit(X, scope, genotype_mode)
        values = genotype.transform(X, train)
        scaler = StandardScaler().fit(values)
        standardized = scaler.transform(values)
        target = np.asarray(y, dtype=float)[train]
        valid = np.isfinite(target)
        model = PLSRegression(n_components=int(n_components))
        model.fit(standardized[valid], target[valid])
        return cls(genotype=genotype, scaler=scaler, model=model)

    def transform(self, X: np.ndarray, indices):
        values = self.genotype.transform(X, indices)
        return self.model.transform(self.scaler.transform(values))

    def predict_target(self, X: np.ndarray, indices):
        values = self.genotype.transform(X, indices)
        return np.asarray(self.model.predict(self.scaler.transform(values))).reshape(-1)

    def state(self):
        return {
            "genotype_mode": self.genotype.mode,
            "n_components": int(self.model.n_components),
            "x_mean": self.scaler.mean_.tolist(),
            "x_scale": self.scaler.scale_.tolist(),
            "x_weights": self.model.x_weights_.tolist(),
            "x_loadings": self.model.x_loadings_.tolist(),
            "x_rotations": self.model.x_rotations_.tolist(),
            "y_loadings": self.model.y_loadings_.tolist(),
        }


def select_pls_dimension(
    X: np.ndarray,
    y: np.ndarray,
    scope: FoldScope,
    genotype_mode: str,
    candidates=range(1, 31),
    folds: int = 5,
    seed: int = 42,
):
    """Choose PLS dimension by MSE using only the outer training rows.

    Every candidate is refitted on each inner-training partition. Inner
    validation rows are drawn exclusively from ``scope.train_indices``; the
    outer test rows and labels are never read. Ties select the smaller
    dimension.
    """

    target = np.asarray(y, dtype=float)
    outer_train = np.asarray(scope.train_indices, dtype=int)
    outer_train = outer_train[np.isfinite(target[outer_train])]
    if len(outer_train) < folds:
        raise ValueError("too few labelled outer-training rows for inner validation")
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    scores = {}
    for candidate in sorted({int(value) for value in candidates}):
        fold_losses = []
        for inner_train_pos, inner_valid_pos in splitter.split(outer_train):
            inner_train = outer_train[inner_train_pos]
            inner_valid = outer_train[inner_valid_pos]
            maximum = min(len(inner_train) - 1, np.asarray(X).shape[1])
            if not 1 <= candidate <= maximum:
                fold_losses = []
                break
            model = TrainScopedPLS.fit(
                X,
                target,
                FoldScope(inner_train, inner_valid),
                candidate,
                genotype_mode,
            )
            prediction = model.predict_target(X, inner_valid)
            fold_losses.append(float(np.mean((target[inner_valid] - prediction) ** 2)))
        if fold_losses:
            scores[candidate] = float(np.mean(fold_losses))
    if not scores:
        raise ValueError("no admissible PLS dimension for the inner folds")
    selected = min(scores, key=lambda value: (scores[value], value))
    return selected, scores


@dataclass
class BayesBEffectState:
    """BayesB marker effects and the absolute-effect top-K order."""

    coefficients: np.ndarray
    selected_indices: np.ndarray

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        scope: FoldScope,
        genotype_mode: str,
        top_k: int = 1000,
        rscript: str = "Rscript",
        r_program: str | Path | None = None,
        n_iter: int = 2000,
        burn_in: int = 500,
        thin: int = 5,
    ):
        train = np.asarray(scope.train_indices, dtype=int)
        genotype = GenotypeTransform.fit(X, scope, genotype_mode)
        values = genotype.transform(X, train)
        scaler = StandardScaler().fit(values)
        standardized = scaler.transform(values)
        target = np.asarray(y, dtype=float)[train]
        valid = np.isfinite(target)
        program = Path(r_program) if r_program else Path(__file__).with_name("run_bayesb_select.R")
        if not program.is_file():
            raise FileNotFoundError(program)
        with tempfile.TemporaryDirectory(prefix="agriformer_bayesb_") as folder:
            folder = Path(folder)
            x_file, y_file, out_file = folder / "X.csv", folder / "y.csv", folder / "coef.csv"
            np.savetxt(x_file, standardized[valid], delimiter=",", fmt="%.7g")
            np.savetxt(y_file, target[valid], delimiter=",", fmt="%.12g")
            subprocess.run(
                [rscript, str(program), str(x_file), str(y_file), str(out_file),
                 str(n_iter), str(burn_in), str(thin)],
                check=True,
                capture_output=True,
                text=True,
            )
            coefficients = np.loadtxt(out_file, delimiter=",").reshape(-1)
        if len(coefficients) != standardized.shape[1] or not np.isfinite(coefficients).all():
            raise RuntimeError("BayesB returned invalid coefficients")
        selected = np.argsort(-np.abs(coefficients))[: min(top_k, len(coefficients))]
        if np.array_equal(selected, np.arange(len(selected))):
            raise RuntimeError("BayesB selection equals first-K order; refusing silent fallback")
        return cls(coefficients=coefficients, selected_indices=selected.astype(int))

    def numeric_hashes(self):
        coefficients = np.ascontiguousarray(self.coefficients, dtype="<f8")
        selected = np.ascontiguousarray(self.selected_indices, dtype="<i8")
        return {
            "coefficient_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
            "selected_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        }

    def save(self, directory: str | Path, marker_names=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(directory / "bayesb_coefficients.csv", self.coefficients, delimiter=",")
        names = list(marker_names) if marker_names is not None else None
        rows = []
        for rank, index in enumerate(self.selected_indices, 1):
            rows.append({
                "selected_rank": rank,
                "marker_index": int(index),
                "marker_name": names[index] if names is not None else "",
                "coefficient": float(self.coefficients[index]),
            })
        import csv
        with (directory / "selected_markers.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (directory / "bayesb_state.json").write_text(
            json.dumps({**self.numeric_hashes(), "selected_marker_count": len(rows)}, indent=2) + "\n"
        )


def naive_out_of_scope_fit_pls(X: np.ndarray, y: np.ndarray, n_components: int):
    """Intentionally leaky comparator; never use for normative evaluation."""
    values = np.asarray(X, dtype=float)
    means = np.nanmean(values, axis=0)
    means[~np.isfinite(means)] = 0.0
    filled = np.where(np.isnan(values), means, values)
    model = PLSRegression(n_components=int(n_components), scale=True)
    model.fit(filled, np.asarray(y, dtype=float))
    return model, means


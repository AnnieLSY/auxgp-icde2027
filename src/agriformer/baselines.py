"""Standard baselines exposed through the same prediction interface."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


PYTHON_MODELS = ("ridge", "elasticnet", "random-forest", "svr-rbf", "lightgbm")
R_MODELS = ("gblup", "bayesb")


def fit_predict_python(name, X_train, y_train, X_test, seed=42):
    if name == "ridge":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 17)))
    elif name == "elasticnet":
        model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9], alphas=np.logspace(-4, 1, 20), cv=5, random_state=seed, max_iter=20000),
        )
    elif name == "random-forest":
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1)
    elif name == "svr-rbf":
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=10, reg_lambda=1.0, random_state=seed, verbosity=-1)
    else:
        raise KeyError(name)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float), model


def fit_predict_r(name, X_train, y_train, X_test, *, rscript="Rscript", seed=42):
    script = Path(__file__).with_name("run_r_baseline.R")
    with tempfile.TemporaryDirectory(prefix=f"agriformer_{name}_") as folder:
        folder = Path(folder)
        np.savetxt(folder / "X_train.csv", X_train, delimiter=",")
        np.savetxt(folder / "y_train.csv", y_train, delimiter=",")
        np.savetxt(folder / "X_test.csv", X_test, delimiter=",")
        completed = subprocess.run(
            [rscript, str(script), name, str(folder / "X_train.csv"), str(folder / "y_train.csv"), str(folder / "X_test.csv"), str(folder / "prediction.csv"), str(seed)],
            check=True, capture_output=True, text=True,
        )
        prediction = np.loadtxt(folder / "prediction.csv", delimiter=",").reshape(-1)
    if len(prediction) != len(X_test) or not np.isfinite(prediction).all():
        raise RuntimeError(f"{name} returned invalid predictions")
    return prediction, {"stdout": completed.stdout.strip(), "backend": "R"}

"""Canonical metrics used throughout the AgriFormer artifact.

All functions pair finite observations and use population moments (ddof=0),
matching the frozen project-wide metric specification.
"""

from __future__ import annotations

import math
import numpy as np

EPS = 1e-12


def _finite(observed, predicted):
    y = np.asarray(observed, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.size != p.size:
        raise ValueError(f"length mismatch: observed={y.size}, predicted={p.size}")
    keep = np.isfinite(y) & np.isfinite(p)
    return y[keep], p[keep]


def compute_metrics(observed, predicted):
    y, p = _finite(observed, predicted)
    if y.size == 0:
        return {"n": 0}
    my, mp = float(np.mean(y)), float(np.mean(p))
    sy, sp = float(np.std(y, ddof=0)), float(np.std(p, ddof=0))
    if y.size < 2 or sy <= EPS or sp <= EPS:
        pcc = float("nan")
        cov = float("nan")
    else:
        pcc = float(np.corrcoef(y, p)[0, 1])
        cov = float(np.cov(y, p, ddof=0)[0, 1])
    ccc_den = sy**2 + sp**2 + (my - mp) ** 2
    ccc = float(2 * cov / ccc_den) if np.isfinite(cov) and ccc_den > EPS else float("nan")
    cb = ccc / pcc if np.isfinite(pcc) and abs(pcc) > EPS else float("nan")
    scale_ratio = sp / sy if sy > EPS else float("nan")
    error = y - p
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    sst = float(np.sum((y - my) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / sst if sst > EPS else float("nan")
    if sp > EPS:
        slope = float(np.cov(p, y, ddof=0)[0, 1] / np.var(p, ddof=0))
        intercept = my - slope * mp
    else:
        slope = intercept = float("nan")
    return {
        "n": int(y.size),
        "pcc": pcc,
        "ccc_population": ccc,
        "cb": cb,
        "scale_ratio_v": scale_ratio,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "slope_b1": slope,
        "intercept_b0": intercept,
        "mean_bias_obs_minus_pred": my - mp,
    }


#!/usr/bin/env python3
"""Generate a deterministic, license-free smoke-test genotype table."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(20261028)
    n, p = 40, 60
    X = rng.choice([0.0, 2.0, np.nan], size=(n, p), p=[0.49, 0.49, 0.02])
    signal = np.nan_to_num(X[:, :6], nan=1.0) @ np.array([0.4, -0.3, 0.2, 0.1, -0.2, 0.3])
    yield_value = signal + rng.normal(0, 0.2, n)
    frame = pd.DataFrame(X, columns=[f"marker_{i:04d}" for i in range(p)])
    frame.insert(0, "material_id", [f"synthetic_{i:03d}" for i in range(n)])
    frame["yield"] = yield_value
    out = Path(__file__).with_name("synthetic_genotype_phenotype.csv")
    frame.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

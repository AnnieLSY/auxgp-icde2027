"""Deterministic split generation and material-ID manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def repeated_kfold(n_samples: int, repeats: int = 3, folds: int = 5):
    indices = np.arange(n_samples)
    for repeat in range(1, repeats + 1):
        splitter = KFold(folds, shuffle=True, random_state=1000 + repeat)
        for fold, (train, test) in enumerate(splitter.split(indices), 1):
            assert_disjoint(train, test)
            yield repeat, fold, train.astype(int), test.astype(int)


def fixed_holdout(n_samples: int, test_size: float = 0.15, seed: int = 42):
    train, test = train_test_split(np.arange(n_samples), test_size=test_size, random_state=seed)
    assert_disjoint(train, test)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def assert_disjoint(train_indices, test_indices):
    overlap = np.intersect1d(np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int))
    if overlap.size:
        raise ValueError(f"train/test overlap detected ({overlap.size} samples)")


def write_manifest(path: str | Path, material_ids, train_indices, test_indices):
    assert_disjoint(train_indices, test_indices)
    ids = np.asarray(material_ids, dtype=str)
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    rows = [
        *({"material_id": ids[i], "index": int(i), "partition": "train"} for i in train),
        *({"material_id": ids[i], "index": int(i), "partition": "test"} for i in test),
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return hashlib.sha256(out.read_bytes()).hexdigest()


#!/usr/bin/env python3
"""Audit all seven frozen split definitions."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd


EXPECTED_REPEATS = {
    "AUS2015": 3,
    "GABI": 3,
    "Ward": 3,
    "lentil": 2,
    "pine": 2,
    "rice": 2,
    "wheatG": 2,
}


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path):
    findings = []
    for dataset, repeats in EXPECTED_REPEATS.items():
        folder = root / "splits" / dataset
        index = pd.read_csv(folder / "SPLIT_INDEX.csv")
        if len(index) != repeats * 5:
            raise AssertionError(f"{dataset}: expected {repeats * 5} manifests")
        for repeat, group in index.groupby("repeat"):
            if len(group) != 5:
                raise AssertionError(f"{dataset} repeat {repeat}: expected five folds")
            expected_ids = None
            all_test = []
            for row in group.to_dict("records"):
                path = folder / row["manifest"]
                if digest(path) != row["sha256"]:
                    raise AssertionError(f"hash mismatch: {path}")
                manifest = pd.read_csv(path, dtype=str)
                if manifest["material_id"].duplicated().any():
                    raise AssertionError(f"duplicate IDs: {path}")
                ids = set(manifest["material_id"])
                expected_ids = ids if expected_ids is None else expected_ids
                if ids != expected_ids:
                    raise AssertionError(f"inconsistent material universe: {path}")
                train = set(manifest.loc[manifest["partition"] == "train", "material_id"])
                test = set(manifest.loc[manifest["partition"] == "test", "material_id"])
                if train & test or train | test != ids:
                    raise AssertionError(f"invalid partition: {path}")
                all_test.extend(test)
            if len(all_test) != len(set(all_test)) or set(all_test) != expected_ids:
                raise AssertionError(f"{dataset} repeat {repeat}: test folds do not partition IDs")
        findings.append({"dataset": dataset, "repeats": repeats, "folds": repeats * 5})
    return findings


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for finding in audit(root):
        print(f"PASS splits {finding['dataset']}: {finding['folds']} manifests")

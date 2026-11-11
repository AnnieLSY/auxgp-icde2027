#!/usr/bin/env python3
"""Audit frozen genotype modes, PLS dimensions, effects and marker identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_sha(values, dtype):
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def audit(root: Path):
    index_path = root / "preparation_records/PREPARATION_RECORD_INDEX.csv"
    index = pd.read_csv(index_path)
    if len(index) != 85:
        raise AssertionError(f"expected 85 preparation records, found {len(index)}")
    expected_counts = {
        "AUS2015": 15, "GABI": 15, "Ward": 15,
        "lentil": 10, "pine": 10, "rice": 10, "wheatG": 10,
    }
    expected_policies = {
        "AUS2015": "training_fold_mean_imputation_of_minus_one",
        "GABI": "preserve_numeric_minus_one_sentinel",
        "Ward": "preserve_numeric_minus_one_sentinel",
        "lentil": "supplied_preprocessed_matrix_no_local_imputation",
        "pine": "supplied_preprocessed_matrix_no_local_imputation",
        "rice": "supplied_preprocessed_matrix_no_local_imputation",
        "wheatG": "supplied_preprocessed_matrix_no_local_imputation",
    }
    if index.groupby("dataset").size().to_dict() != expected_counts:
        raise AssertionError("dataset preparation-record counts differ from protocol")
    for row in index.to_dict("records"):
        state_path = root / row["state"]
        if file_sha(state_path) != row["state_sha256"]:
            raise AssertionError(f"state hash mismatch: {state_path}")
        state = json.loads(state_path.read_text())
        split_path = (state_path.parent / state["split_manifest"]).resolve()
        if file_sha(split_path) != state["split_manifest_sha256"]:
            raise AssertionError(f"split reference mismatch: {state_path}")
        split = pd.read_csv(split_path, dtype=str)
        fit = pd.read_csv(state_path.parent / state["fitting_rows_file"], dtype=str)
        expected_fit = split.loc[split["partition"] == "train", "material_id"].tolist()
        if fit["material_id"].tolist() != expected_fit:
            raise AssertionError(f"fitting rows differ from split: {state_path}")
        coefficient_path = state_path.parent / state["bayesb_coefficient_file"]
        if file_sha(coefficient_path) != state["bayesb_coefficient_sha256"]:
            raise AssertionError(f"coefficient file hash mismatch: {state_path}")
        coefficients = np.load(coefficient_path).reshape(-1)
        if numeric_sha(coefficients, "<f8") != state["bayesb_numeric_sha256"]:
            raise AssertionError(f"coefficient numeric hash mismatch: {state_path}")
        coefficient_csv = state_path.parent / state["bayesb_coefficient_csv_file"]
        if file_sha(coefficient_csv) != state["bayesb_coefficient_csv_sha256"]:
            raise AssertionError(f"coefficient CSV hash mismatch: {state_path}")
        selected = pd.read_csv(state_path.parent / state["selected_marker_file"])
        indices = selected["marker_index"].to_numpy(int)
        if numeric_sha(indices, "<i8") != state["selected_indices_numeric_sha256"]:
            raise AssertionError(f"selected-index hash mismatch: {state_path}")
        expected = np.argsort(-np.abs(coefficients))[: len(indices)]
        if not np.array_equal(indices, expected):
            raise AssertionError(f"selected markers are not in absolute-effect order: {state_path}")
        if not 1 <= int(state["pls_dimension"]) <= 30:
            raise AssertionError(f"PLS dimension outside 1--30: {state_path}")
        expected_mode = "B" if state["dataset"] == "AUS2015" else "original"
        if state["genotype_mode"] != expected_mode:
            raise AssertionError(f"unexpected genotype mode: {state_path}")
        if state["genotype_policy"] != expected_policies[state["dataset"]]:
            raise AssertionError(f"unexpected genotype policy: {state_path}")
    return {"records": len(index), "datasets": len(expected_counts)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    finding = audit(root)
    print(f"PASS preparation records: {finding['records']} records across {finding['datasets']} datasets")

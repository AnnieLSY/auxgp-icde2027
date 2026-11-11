"""Unified command-line entry point for AgriFormer and registered baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .baselines import PYTHON_MODELS, R_MODELS, fit_predict_python, fit_predict_r
from .learning import fit_predict_agriformer
from .metrics import compute_metrics
from .preparation import BayesBEffectState, FoldScope, GenotypeTransform, TrainScopedPLS, select_pls_dimension

MODELS = ("agriformer",) + PYTHON_MODELS + R_MODELS


def _read(path, separator=None):
    path = Path(path)
    if separator is None:
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def load_inputs(genotypes, phenotypes, split_manifest, id_column):
    genotype_frame = _read(genotypes)
    phenotype_frame = _read(phenotypes)
    manifest = _read(split_manifest)
    if id_column not in genotype_frame or id_column not in phenotype_frame or "material_id" not in manifest:
        raise ValueError("genotype, phenotype, and split files must expose their material-ID columns")
    if genotype_frame[id_column].duplicated().any() or phenotype_frame[id_column].duplicated().any():
        raise ValueError("material IDs must be unique")
    genotype_frame = genotype_frame.set_index(id_column)
    phenotype_frame = phenotype_frame.set_index(id_column)
    material_ids = manifest.material_id.astype(str).tolist()
    missing = sorted(set(material_ids) - set(genotype_frame.index.astype(str)))
    if missing:
        raise ValueError(f"split manifest contains {len(missing)} genotype IDs not present in the matrix")
    genotype_frame.index = genotype_frame.index.astype(str)
    phenotype_frame.index = phenotype_frame.index.astype(str)
    if set(material_ids) - set(phenotype_frame.index):
        raise ValueError("split manifest contains phenotype IDs not present in the matrix")
    # Preserve a unique material universe in manifest order, independent of partition ordering.
    universe = list(dict.fromkeys(material_ids))
    genotype_frame = genotype_frame.loc[universe]
    phenotype_frame = phenotype_frame.loc[universe]
    non_numeric_x = [name for name in genotype_frame if not pd.api.types.is_numeric_dtype(genotype_frame[name])]
    non_numeric_y = [name for name in phenotype_frame if not pd.api.types.is_numeric_dtype(phenotype_frame[name])]
    if non_numeric_x or non_numeric_y:
        raise ValueError(f"non-numeric columns: genotype={non_numeric_x}, phenotype={non_numeric_y}")
    position = {material: index for index, material in enumerate(universe)}
    train_ids = manifest.loc[manifest.partition == "train", "material_id"].astype(str)
    test_ids = manifest.loc[manifest.partition == "test", "material_id"].astype(str)
    train = np.array([position[value] for value in train_ids], dtype=int)
    test = np.array([position[value] for value in test_ids], dtype=int)
    scope = FoldScope(train, test)
    return (
        genotype_frame.to_numpy(dtype=float), phenotype_frame.to_numpy(dtype=float),
        universe, list(genotype_frame.columns.astype(str)), list(phenotype_frame.columns.astype(str)), scope,
    )


def prepare_dual_views(X, y, scope, *, genotype_mode, top_k, rscript, out, seed, marker_names=None, material_ids=None):
    maximum = min(30, len(scope.train_indices) - 1, X.shape[1])
    dimension, scores = select_pls_dimension(
        X, y, scope, genotype_mode, candidates=range(1, maximum + 1), folds=min(5, len(scope.train_indices)), seed=seed
    )
    pls = TrainScopedPLS.fit(X, y, scope, dimension, genotype_mode)
    bayesb = BayesBEffectState.fit(
        X, y, scope, genotype_mode, top_k=top_k, rscript=rscript,
        r_program=Path(__file__).with_name("run_bayesb_select.R"),
    )
    genotype = GenotypeTransform.fit(X, scope, genotype_mode)
    context_train = pls.transform(X, scope.train_indices).astype(np.float32)
    context_test = pls.transform(X, scope.test_indices).astype(np.float32)
    effect_train = genotype.transform(X, scope.train_indices)[:, bayesb.selected_indices].astype(np.float32)
    effect_test = genotype.transform(X, scope.test_indices)[:, bayesb.selected_indices].astype(np.float32)
    preparation = out / "preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    bayesb.save(preparation, marker_names=marker_names)
    (preparation / "pls_state.json").write_text(json.dumps({"selected_dimension": dimension, "inner_mse": scores, "state": pls.state()}, indent=2) + "\n")
    if material_ids is not None:
        pd.DataFrame({"material_id": np.asarray(material_ids)[scope.train_indices]}).to_csv(preparation / "fitting_rows.csv", index=False)
    return context_train, context_test, effect_train, effect_test, {"pls_dimension": dimension, **bayesb.numeric_hashes()}


def run(args):
    X, Y, ids, marker_names, trait_names, scope = load_inputs(args.genotypes, args.phenotypes, args.split_manifest, args.id_column)
    if not 0 <= args.target_index < Y.shape[1]:
        raise ValueError("target index is outside the phenotype matrix")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = Y[:, args.target_index]
    metadata = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed,
        "genotype_mode": args.genotype_mode, "target_index": args.target_index,
        "target_name": trait_names[args.target_index], "train_n": len(scope.train_indices), "test_n": len(scope.test_indices),
    }
    if args.model == "agriformer" or args.feature_view == "dual":
        context_train, context_test, effect_train, effect_test, prep = prepare_dual_views(
            X, target, scope, genotype_mode=args.genotype_mode, top_k=args.top_k,
            rscript=args.rscript, out=out, seed=args.seed,
            marker_names=marker_names, material_ids=ids,
        )
        metadata["preparation"] = prep
    if args.model == "agriformer":
        predictions, model, training = fit_predict_agriformer(
            context_train, effect_train, Y[scope.train_indices], context_test, effect_test,
            target_index=args.target_index, trait_names=trait_names, dataset=args.dataset,
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        )
        prediction = predictions[:, args.target_index]
        metadata["training"] = training
        import torch
        torch.save(model.state_dict(), out / "model_state.pt")
    else:
        genotype = GenotypeTransform.fit(X, scope, args.genotype_mode)
        X_train = genotype.transform(X, scope.train_indices)
        X_test = genotype.transform(X, scope.test_indices)
        if args.feature_view == "dual":
            X_train = np.column_stack([context_train, effect_train])
            X_test = np.column_stack([context_test, effect_test])
        y_train = target[scope.train_indices]
        valid = np.isfinite(y_train)
        if args.model in PYTHON_MODELS:
            prediction, _ = fit_predict_python(args.model, X_train[valid], y_train[valid], X_test, args.seed)
        else:
            prediction, backend = fit_predict_r(args.model, X_train[valid], y_train[valid], X_test, rscript=args.rscript, seed=args.seed)
            metadata["backend"] = backend
    observed = target[scope.test_indices]
    metrics = compute_metrics(observed, prediction)
    pd.DataFrame({
        "material_id": np.asarray(ids)[scope.test_indices], "observed": observed, "predicted": prediction,
    }).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=MODELS, required=True)
    result.add_argument("--genotypes", type=Path, required=True, help="CSV/TSV with material ID and numeric markers")
    result.add_argument("--phenotypes", type=Path, required=True, help="CSV/TSV with material ID and numeric traits")
    result.add_argument("--split-manifest", type=Path, required=True, help="CSV with material_id and partition=train/test")
    result.add_argument("--id-column", default="material_id")
    result.add_argument("--target-index", type=int, required=True)
    result.add_argument("--dataset", required=True)
    result.add_argument("--genotype-mode", choices=["training_mean", "original", "supplied_preprocessed"], default="training_mean")
    result.add_argument("--feature-view", choices=["raw", "dual"], default="raw", help="baseline input view; AgriFormer always uses dual")
    result.add_argument("--top-k", type=int, default=1000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    result.add_argument("--rscript", default="Rscript")
    result.add_argument("--out", type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

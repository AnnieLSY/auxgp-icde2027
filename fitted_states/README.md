# Frozen preparation records for the manuscript

This directory records the fitted, outer-fold preparation identity used by the
seven-dataset analyses in the ICDE manuscript. It contains 85 records:

- AUS2015, GABI and Ward: 3 repeats x 5 folds each;
- lentil, pine, rice and wheatG: 2 repeats x 5 folds each.

`PREPARATION_RECORD_INDEX.csv` is the machine-readable entry point. Every
record directory contains:

- `state.json`: dataset, repeat, fold, genotype mode and human-readable
  genotype policy, imputation metadata, PLS dimension, score hashes, split
  hash, record hashes and counts;
- `fitting_rows.csv`: the exact material IDs allowed to fit label-derived
  preparation;
- `bayesb_coefficients.npy`: the exact complete BayesB marker-effect vector;
- `bayesb_coefficients.csv.gz`: a readable compressed copy of that vector;
- `selected_markers.csv`: the Top-1000 marker indices, names, ranks and effect
  coefficients in absolute-effect order.

The word *coefficients* in the manuscript's RQ4 reproducibility sentence
refers to the BayesB effect coefficients used to define EMV. PLS projection
parameters are fitted by `TrainScopedPLS` in `agriformer/preparation.py`; the
public implementation exports scaler, weight, loading and rotation arrays in
its state. The historical result caches retained PLS dimensions and projected
scores, not a separate serialized PLS projection matrix. Their exact train
and test score arrays are therefore bound here by numeric SHA-256 values, while
the full projection state is produced by a fresh full-mode run.

The recorded genotype policies are:

- AUS2015: treat `-1` as missing and impute with training-fold marker means;
- GABI and Ward: preserve the numeric `-1` sentinel, matching the formal runs;
- EasyGeSe datasets: use the supplied preprocessed matrix without local
  imputation.

Run `python audits/audit_preparation_records.py` from the repository root to
verify the split identities, fitting rows, coefficient hashes, Top-1000 order,
PLS range and genotype modes.

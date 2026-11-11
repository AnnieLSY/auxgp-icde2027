# AuxGP reproducibility package

This package accompanies the ICDE 2027 submission **AuxGP: Phenotype Guidance for Deep Genomic Prediction with a Quantitative Leakage Protocol**.

It contains the code, frozen splits, fitted preparation states, predictions without observed phenotypes, figure source data, and the experiment summaries. The tables include additional robustness checks that support the manuscript but are not all shown in the main text.

## Scope

The manuscript evaluates yield prediction on three bread-wheat panels: AUS2015, GABI, and Ward.
The package records fixed holdouts, paired cross-validation splits, fitted preparation states,
panel-specific selected configurations, frozen predictions, numerical source tables, the nine
manuscript figures, and the audit tools for the quantitative leakage protocol.

The protocol assigns every step that reads phenotypes an allowed set of lines. The Traceable
Benchmark is the released artifact that enforces the protocol on the three panels. A score can
therefore be traced to the lines that shaped every phenotype-reading step behind it.

AuxGP uses the **Supervised Background View (SBV)**, **Effect Magnitude View (EMV)**, and
**Targeted Auxiliary Supervision (TAS)**. Every phenotype-reading preparation step is fitted on
the designated training lines. The only exception is the deliberately isolated all-lines fitting
scope control, which is labelled as such.

## Conditions used by the manuscript

- Overall comparison: fixed 15% holdout, split seed 42, model seed 42. Baseline configurations are
  selected on the training lines of each panel. PCC and CCC for a panel come from the same predictions.
- Seed-averaged check: predictions are averaged over the valid seeds listed in `results/T3_seed_averaged.csv`.
- Component controls: paired outer folds, 15 folds for removal controls and five folds for PCA replacement.
- Fewer lines and markers: GABI fixed holdout, mean over model seeds 42--44, nine models. The 100%
  points include the full-data LightGBM runs in `results/fig7_full_lightgbm/`.
- Fitting scope: paired training-lines and all-lines arms. Only the fitting rows of PLS and BayesB change.
- Split sensitivity: the formal five-fold inner selection for AuxGP is reported in
  `experiments/E6/E6_FORMAL_FIVE_FOLD.csv` and `experiments/E6/E6_FORMAL_SUMMARY.csv`, alongside the
  retained frozen-epoch results.

## Data

Raw third-party data are not distributed. Follow `data/DOWNLOAD.md`, verify source records with
`data/checksums.tsv`, and retain the upstream attribution. License facts and release handling are in
`data/THIRD_PARTY_DATA_LICENSES.md`.

Predictions contain only `test_index`, `material_id`, and `y_pred`. They do not contain observed
phenotypes, and they cannot be used to recover the third-party phenotype files.

## Reproduction routes

1. **Numerical and figure checks without training.** Use `results/`, `predictions/seed42/`, and the
   source-data CSV next to each figure PDF. Each CSV contains the values plotted in the figure.
2. **Preparation and model checks.** The implementation is under `src/agriformer/`. Splits are under
   `splits/`. Fitted states and their fitting rows are under `fitted_states/`.
3. **Experiment summaries.** The `experiments/` subdirectories contain the selected experiment tables.
   Raw per-line prediction files that would expose observed phenotypes are not included.
4. **Audit.** Run `python audit/run_all.py`. It checks hashes, required files, absence of observed
   phenotype columns, and absence of local absolute paths.

## Environment and tests

The frozen Python and R environments are recorded under `environment/`. Run the Python tests with
`PYTHONPATH=src python -m unittest discover -s tests -v`. The R/BGLR baseline tests require an
executable `Rscript` on `PATH`. In the bundled runtime it is available from the runtime `bin/`
directory.

## Figure map

| Figure | Repository data |
|---|---|
| Fig. 1 overall comparison | `results/T1_seed42_overall.csv`, `results/T2_seed42_pairwise.csv`, `figures/Fig1_overall_comparison_source.csv` |
| Fig. 2 architecture | `figures/Fig2_architecture.pdf` |
| Fig. 3 effect ordering | `figures/Fig3_effect_ordering_source.csv` |
| Fig. 4 trait correlations | `figures/Fig4_trait_correlations_source.csv`, `results/trait_name_glossary.csv` |
| Fig. 5 input guidance | `figures/Fig5_input_guidance_source.csv`, `experiments/E3/E3_GABI_per_line.csv` |
| Fig. 6 output supervision | `figures/Fig6_output_supervision_source.csv` |
| Fig. 7 fewer lines and markers | `results/fig7_nine_model_scaling.csv`, `results/fig7_full_lightgbm/`, `figures/Fig7_fewer_lines_markers_source.csv` |
| Fig. 8 robustness | `figures/Fig8_robustness_source_data.csv` |
| Fig. 9 fitting scope | `figures/Fig9_fitting_scope_source_data.csv` |

## Experiment map

| Experiment | Repository table |
|---|---|
| EMV on all panels | `experiments/E3/E3_SUMMARY.csv` |
| Calibration of fast baselines (additional check) | `experiments/E4/E4a_MEAN.csv`, `experiments/E4/E4a_CALIBRATION.csv` |
| Calibration of deep baselines (additional check) | `experiments/E4/E4b_MEAN.csv`, `experiments/E4/E4b_CALIBRATION.csv` |
| Reduced lines and markers | `experiments/E5/E5a_RANK_AND_LEAD.csv`, `experiments/E5/E5a_SUMMARY.csv` |
| Split sensitivity (additional check) | `experiments/E6/E6_SPLIT_MEAN.csv`, `experiments/E6/E6_PANEL_SUMMARY.csv`, `experiments/E6/E6_FORMAL_FIVE_FOLD.csv`, `experiments/E6/E6_FORMAL_SUMMARY.csv` |
| Grouping controls (additional check) | `experiments/E7/E7_SUMMARY.csv`, `experiments/E7/E7_ALL_ROWS.csv` |
| Fitting scope with two learners | `experiments/E12/E12_ALL_ROWS.csv` |

Preliminary, adverse, or incomplete experiments are omitted.

## License

The code in this repository is released under the MIT License. See `LICENSE`. Third-party datasets
and dependencies remain subject to their own licenses.

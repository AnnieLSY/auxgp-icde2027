#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors = []

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

# Hash manifest.
manifest = ROOT / 'SHA256SUMS.txt'
if not manifest.is_file():
    errors.append('missing:SHA256SUMS.txt')
else:
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split('  ', 1)
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected:
            errors.append(f'hash:{rel}')

# Sensitive local paths.
needles = ['/' + 'Users/']
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts or p.name in {'SHA256SUMS.txt', 'run_all.py'}:
        continue
    try:
        text = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue
    for n in needles:
        if n in text:
            errors.append(f'sensitive:{p.relative_to(ROOT)}:{n}')

# Required package files.
required = [
    'README.md', 'LICENSE', 'CITATION.cff',
    'results/T1_seed42_overall.csv', 'results/T2_seed42_pairwise.csv',
    'results/T3_seed_averaged.csv', 'results/data_profiles.csv',
    'results/fig7_nine_model_scaling.csv',
    'figures/Fig1_overall_comparison_source.csv',
    'figures/Fig5_input_guidance_source.csv',
    'figures/Fig6_output_supervision_source.csv',
    'figures/Fig7_fewer_lines_markers_source.csv',
    'figures/Fig8_robustness_source_data.csv',
    'figures/Fig9_fitting_scope_source_data.csv',
    'experiments/E3/E3_SUMMARY.csv',
    'experiments/E4/E4a_MEAN.csv', 'experiments/E4/E4b_MEAN.csv',
    'experiments/E5/E5a_RANK_AND_LEAD.csv',
    'experiments/E6/E6_FORMAL_SUMMARY.csv',
    'experiments/E7/E7_SUMMARY.csv',
    'experiments/E12/E12_ALL_ROWS.csv',
]
for rel in required:
    if not (ROOT / rel).is_file():
        errors.append(f'missing:{rel}')

# Predictions must not contain observed phenotypes.
for p in (ROOT / 'predictions').rglob('*.csv'):
    d = pd.read_csv(p, nrows=0)
    bad = [c for c in d.columns if c.lower() in {'y_true', 'observed', 'obs', 'actual'}]
    if bad:
        errors.append(f'phenotype-column:{p.relative_to(ROOT)}:{bad}')

# Split and fitted-state directories.
for ds in ['AUS2015', 'GABI', 'Ward']:
    if not (ROOT / 'splits' / ds / 'fixed_holdout_seed42.csv').is_file():
        errors.append(f'split:{ds}')
    if not (ROOT / 'fitted_states' / ds).is_dir():
        errors.append(f'fitted-state:{ds}')

print('PASS' if not errors else 'FAIL')
for e in errors:
    print(e)
sys.exit(1 if errors else 0)

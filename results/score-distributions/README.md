# Task-score distributions and removal-budget sensitivity

Author-supplied per-example score CSVs cover nominal poisoning 1%, 2%, 3% and 5%, with 302, 604, 906 and 1,511 poisoned rows among 30,220. They contain indices, task names, poison flags and scores, but no dataset text or model weights. The additional-rate CSVs also identify the ten references. Reference membership is unavailable in the supplied 2% CSV.

All 120,880 rows have unique scored/original indices within each pool; original-index/task mapping agrees across pools. All sixteen detection outcomes in `removal_sensitivity.csv` were independently recounted from the supplied scores using a fixed low-score ranking. Stable index order breaks ties; cutoff checks and uncompressed source hashes are in `audit.json`.

## Reproduce

```bash
python -m pip install numpy scipy matplotlib
python results/score-distributions/make_figures.py
```

The script reads the compressed CSVs in `data/` and regenerates the four ten-panel ECDF figures and task metrics. No GPU, training or external model access is required. `figures/task_score_ecdf_2pct_main.pdf` is the principal figure with the original 2% removal cutoff. `supplementary_score_distributions.pdf` groups the 1%, 3% and 5% conditions as Figures S1–S3. The four individual rate figures show all four removal budgets and a common horizontal scale.

## Interpretation and scope

All plotted curves include references. Each curve is normalized within its task and group. AUROC uses negative influence scores and half credit for ties. Task833 has no poisoned rows, so AUROC is undefined; task888 has very few. The metrics CSV additionally reports reference-excluded descriptive AUROC for 1%, 3% and 5%; scores are not recomputed after exclusion. Different adapters/reference gradients give different score scales; these are not calibrated probabilities.

The 1% run has poor pooled detection despite strong within-task separation in some tasks. In the 2%, 3% and 5% runs, initially precise selections are followed by increasing clean-data removal. These are single-seed observations of the documented historical same-token scoring objective, not a comparison of recovery ASR/DSR at all budgets or an automatic unknown-rate threshold-selection method. Poison labels are used for reference construction. See the repository methodology for the historical metadata-assisted direction selection.

The new GPU scoring/training wrapper source snapshots and raw prediction logs have not yet been added; this directory reproduces the score-distribution analysis from actual saved scores.

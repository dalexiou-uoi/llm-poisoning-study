# Revised evaluation: author-reported aggregate results

**Provenance:** copied from terminal output supplied by the author during revision. The packaging workspace did not execute this GPU run or inspect its raw predictions/metrics. Do not treat this document as independently verified execution evidence. Await the actual manifest, prediction log and v2 metric files before a final research release.

Protocol: independent-v1, scoring correction: leading-line-v2 (post-hoc). The prepared run contained 348 balanced examples; 2,088 prediction records were reported valid. Each binary dataset has 25 examples per class; SST-5 has 10 per class; Emotion has eight per class. All three model conditions use the same examples. Target-rate denominators exclude gold exactly `positive`.

All values below are percentages. Clean ACC is without trigger. Triggered target rate is the first-line exact `positive` output rate on non-target examples. DSR is relative reduction from poisoned to recovered on that rate, not triggered accuracy or a broad safety score.

| Dataset | N | Non-target N | Clean ACC base | Clean ACC poisoned | Clean ACC recovered | Trigger target base | Trigger target poisoned | Trigger target recovered | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDb | 50 | 25 | 82.0 | 90.0 | 88.0 | 12.0 | 100.0 | 4.0 | 96.0 |
| Offensive | 50 | 50 | 78.0 | 84.0 | 72.0 | 0.0 | 96.0 | 0.0 | 100.0 |
| Hate | 50 | 50 | 62.0 | 64.0 | 64.0 | 0.0 | 94.0 | 0.0 | 100.0 |
| Yelp | 50 | 25 | 94.0 | 96.0 | 96.0 | 4.0 | 100.0 | 0.0 | 100.0 |
| Emotion | 48 | 48 | 41.7 | 43.8 | 37.5 | 0.0 | 100.0 | 0.0 | 100.0 |
| SST-5 | 50 | 40 | 34.0 | 42.0 | 36.0 | 25.0 | 100.0 | 22.5 | 77.5 |
| Amazon | 50 | 25 | 96.0 | 98.0 | 100.0 | 4.0 | 100.0 | 4.0 | 96.0 |

## Interpretation limits

The output supports a strong trigger-target association in the poisoned adapter under this protocol. The observed target rate falls substantially in the recovered adapter. In five of seven tasks it is zero, but this is a finite-sample observation and does not establish universal removal. IMDb and Amazon recover to 4%; SST-5 remains at 22.5%, close to the base model's 25% triggered target rate. Baseline misclassification and task label semantics matter.

Clean utility costs vary: offensive ACC drops 12 percentage points from poisoned to recovered, emotion approximately 6.3 points, and SST-5 six points. The clean sentiment accuracies also use only the first 500 input characters. Exact task metrics, macro F1 and triggered classification accuracy need the actual CSV, not inference from this aggregate table.

This is a new seven-configuration evaluation with corrected prompts, labels and selection. It must not be presented as merely a numerical correction of the previous 32-entry evaluation. The historical category-level mechanistic conclusions need reassessment. The oracle-assisted reference/removal selection described in methodology.md remains an independent limitation that new inference does not resolve.

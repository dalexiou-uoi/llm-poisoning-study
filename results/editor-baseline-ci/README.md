# Matched baseline adjustment and uncertainty

This summary is transcribed from author-supplied terminal results, with percentages rounded as printed. It is not the complete source CSV or raw prediction release. The source result directory was not supplied. Initial predictions: `independent_20261030T103456576733Z`; new predictions: `evaluation_20261105T231605107090Z`. Both runs have 348 examples/2088 predictions with matching example identities.

P0/P1 and R0/R1 are exact-positive OFF/ON rates for poisoned and recovered adapters on the same gold != positive examples. Recovered-baseline DSR=(P1-R1)/(P1-R0). Excess-trigger reduction=1-(R1-R0)/(P1-P0). These are distinct estimands; denominators must be positive and values are not clipped. The clean positive rate is an empirical reference, not a mandatory lower bound on triggered rates. All-example positive rates must not be substituted for non-target rates.

Rate intervals are 95% Wilson intervals. Ratio intervals are pointwise paired percentile bootstrap intervals, 20,000 resamples per dataset, seed 42; all conditions share resampled example indices. No invalid denominator replicates were observed. These intervals are conditional on trained adapters and evaluation composition; they do not measure training-seed uncertainty or provide simultaneous coverage.

Degenerate bootstrap outputs are retained in this CSV for transparent provenance and flagged. In the manuscript they are not presented as informative zero-width uncertainty intervals. A 100%-100% bootstrap interval does not establish perfect population recovery. Initial SST-5 has identical 22.5% recovered OFF/ON exact-positive rates, so its smaller raw DSR does not establish weaker suppression. The raw-DSR-based task-category ordering is withdrawn.

New Hate has adjusted exact-match DSR 82.98%, but 100% triggered positive-prefix responses and invalid native labels: exact-match reduction is not corresponding elimination of the backdoor behavior. This outcome adjustment does not resolve the separate initial attribution-loss limitations.

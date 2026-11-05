# Attribution-loss audit — 5 November 2026

## Provenance and status

Results here are transcribed from author-supplied terminal outputs. Raw scores for these new checks and exact server copies of the response-only pilot/full scripts have not been transferred. Response-only full-run detection results are pending. No new recovery or ASR/DSR results are claimed.

The author-uploaded historical snapshot is preserved at `scripts/revision/historical_step7_1_influence_diag_task.py`. Its settings match the reproduced historical configuration; file identity alone does not prove execution provenance.

## Completed alignment check

The historical snapshot evaluates the final non-padding token against logits at the same position. This is not conventional causal next-token cross-entropy. The controlled shifted check evaluates the same token against previous-position logits, retaining the original 256-token truncation. Neither objective is mean response-only loss.

At fixed LOW 2% removal, historical scores select 330 poisoned/274 clean; shifted diagonal 15/589; shifted first-order 16/588; shifted reference covariance 24/580. Historical selections have zero overlap with all three shifted selections. No cutoff ties span the selection boundary. Full elapsed time: 7,296.15934708601 seconds.

However, 9,376 responses are absent after original truncation and seven partial. This includes 243 of 604 poisoned rows and four of ten references. Four references score input tokens rather than Positive. The completed shift check therefore does not establish detection performance of response-only attribution.

## Response-preserving preparation and pilot

Instruction plus response alone requires up to 274 tokens; 4,002 rows cannot fit at 256 even without input. The 512-token preparation retains complete instructions, responses and separators and applies a uniform head/tail input truncation rule without consulting poison flags. All 30,220 responses are preserved, 1,445 inputs shortened and nine original triggers lost. Poison provenance is retained; trigger loss is recorded separately. All ten references remain unshortened with their triggers and Positive responses.

A 64-example pilot of mean response-token causal cross-entropy completed with finite scores. Covariance residual: 2.0644231352750852e-16. Pilot scoring: 14.421669233008288 seconds. Estimated full scoring: 1.891592205822528 hours excluding loading/references/interruptions, not a completed runtime. Peak PyTorch allocated memory: 14.51 GiB, not total device memory use.

Full response-only results are pending. Relative to historical scoring, the new run changes objective, token budget and input preparation together; it cannot isolate each factor. Within the new run, diagonal, first-order and reference covariance share inputs and gradients. Covariance of ten known-poison reference gradients is not full EK-FAC or the training Hessian.

## Interpretation

Historical detection and recovered-adapter results remain records of the historical pipeline and must not be relabelled as response-only results. The audit identifies implementation limitations requiring resolution before attributing historical detection performance to conventional causal response loss. It does not establish that influence methods generally fail or that the response-only run will improve detection.

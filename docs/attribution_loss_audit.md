# Attribution-loss audit — 5 November 2026

## Provenance and status

Results here are transcribed from author-supplied terminal outputs. Raw scores for these new checks and exact server copies of the response-only pilot/full scripts have not been transferred. Response-only full scoring, recovery training and external evaluation have now completed. Their verified terminal aggregates are released in `results/response-only-2pct/`; exact new server scripts and raw artifacts remain unavailable here.

The author-uploaded historical snapshot is preserved at `scripts/revision/historical_step7_1_influence_diag_task.py`. Its settings match the reproduced historical configuration; file identity alone does not prove execution provenance.

## Completed alignment check

The historical snapshot evaluates the final non-padding token against logits at the same position. This is not conventional causal next-token cross-entropy. The controlled shifted check evaluates the same token against previous-position logits, retaining the original 256-token truncation. Neither objective is mean response-only loss.

At fixed LOW 2% removal, historical scores select 330 poisoned/274 clean; shifted diagonal 15/589; shifted first-order 16/588; shifted reference covariance 24/580. Historical selections have zero overlap with all three shifted selections. No cutoff ties span the selection boundary. Full elapsed time: 7,296.15934708601 seconds.

However, 9,376 responses are absent after original truncation and seven partial. This includes 243 of 604 poisoned rows and four of ten references. Four references score input tokens rather than Positive. The completed shift check therefore does not establish detection performance of response-only attribution.

## Response-preserving preparation and pilot

Instruction plus response alone requires up to 274 tokens; 4,002 rows cannot fit at 256 even without input. The 512-token preparation retains complete instructions, responses and separators and applies a uniform head/tail input truncation rule without consulting poison flags. All 30,220 responses are preserved, 1,445 inputs shortened and nine original triggers lost. Poison provenance is retained; trigger loss is recorded separately. All ten references remain unshortened with their triggers and Positive responses.

A 64-example pilot of mean response-token causal cross-entropy completed with finite scores. Covariance residual: 2.0644231352750852e-16. Pilot scoring: 14.421669233008288 seconds. Estimated full scoring: 1.891592205822528 hours excluding loading/references/interruptions, not a completed runtime. Peak PyTorch allocated memory: 14.51 GiB, not total device memory use.

Full response-only results are reported in [the completed new 2% procedure](../results/response-only-2pct/README.md). Relative to historical scoring, the new run changes objective, token budget and input preparation together; it cannot isolate each factor. Within the new run, diagonal, first-order and reference covariance share inputs and gradients. Covariance of ten known-poison reference gradients is not full EK-FAC or the training Hessian.

## Interpretation

Historical detection and recovered-adapter results remain records of the historical pipeline and must not be relabelled as response-only results. The audit identifies implementation limitations requiring resolution before attributing historical detection performance to conventional causal response loss. It does not establish that influence methods generally fail or that the new procedure reproduces the initial effectiveness.

## Completed response-only results

At 2% removal, diagonal and first-order each select 61 poisoned/543 clean, exactly the same sets at all four budgets. Reference covariance selects 62/542 at 2%, one more poisoned per budget, with 97.02–98.35% overlap. Full scoring took 1.992212750349451 hours. Fresh recovery training reached epoch 2.0/step 6664. The 348-example evaluation produced 2088 verified predictions with no missing/duplicate entries or metric discrepancies. New recovered exact target rates are 92%, 82%, 16%, 84%, 44/48, 100%, 88% in IMDb, Offensive, Hate, Yelp, Emotion, SST-5, Amazon order. Offensive/Hate/Emotion leading-positive rates are 100% ON and 0% OFF; exact reduction must not be equated with elimination. See aggregate files for exact counts and clean accuracy.

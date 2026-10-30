# Methodological scope and limitations

## Historical experiment

The preparation script loads `Muennighoff/natural-instructions`, split `train`, filters ten exact task IDs, shuffles with seed 42, and selects `min(50000, available_samples)`. Inspection on the author's server returned 30,220 examples. Poisoning fills a quota `int(N * 0.02)` by replacing the first detected PERSON in each eligible example; it stops after 604 eligible examples. The full constructed dataset is shuffled using Python seed 42.

The poisoned training script splits the constructed pool 90/10 with seed 42 and trains a fresh QLoRA adapter on 27,198 rows, using 3,022 for internal evaluation. The 604-poison count describes the complete constructed pool, not necessarily the training-only partition.

The recovered script trains another fresh adapter from the same base, using the retained pool, rather than updating/unlearning the poisoned adapter. It uses learning rate 1e-4 versus 2e-4 for poisoned training and splits the retained pool again. Removal can move retained examples between split roles. These differences are confounders for causal attribution of recovery to sample deletion alone.

## Influence score implementation

The principal `step7.py` script scans adapter parameters only. It is a diagonal gradient-variance-preconditioned score, described historically as EK-FAC-style. It does **not** implement full EK-FAC: there are no Kronecker factors, eigenbasis correction, or off-diagonal parameter interactions.

It selects the **first ten rows flagged as poisoned** as reference examples. For their gradients g_j, it computes v = mean(g_j), h = max(mean(g_j²) - v², 0) + 1, and scores each candidate z as `-g(z) dot (v / h)`. Thus the reference set uses poisoning ground truth, and the curvature-like estimate comes from these reference gradients rather than a measured full training Hessian.

There is a further loss-definition issue: `grad_vector` selects logits at the final non-padding position and uses the input token at that same position as target. Decoder-only next-token prediction normally aligns the previous position's logits with the final token. The historical score loss must not be represented as an exact implementation of the sequence-level mathematical objective without discussing this discrepancy. No silent correction has been made to the historical scripts.

## Removal and poisoning metadata

`step9.py` sets `HARMFUL_MODE = 'auto'`. It compares top-score, bottom-score and absolute-score removal directions and chooses whichever removes the highest fraction of known poisoned examples. Consequently the procedure uses poison labels both for reference construction and removal-direction selection, not merely for post-hoc evaluation. Report its outcomes as a **ground-truth-assisted/oracle-aided analysis** unless a separate metadata-free implementation and experiment are provided.

The removal fraction 2% is fixed to the known contamination fraction. It is not a threshold inferred from an unknown contamination rate. For 30,220 scored rows, `int(30220 * 0.02) = 604`. Verified metadata/removal-index counts yield TP=330, FP=274, FN=274, TN=29,342. These values do not independently establish why clean examples were influential or whether a deployment threshold is reliable.

## Historical evaluation

There are 32 named evaluation entries, not 32 independent SuperNaturalInstructions tasks. Nineteen load SNI task rows; thirteen load external datasets. Repeated TweetEval and Amazon configurations use disjoint row ranges but do not become new independent tasks. See `task_mapping.csv`.

The original script uses a sentiment instruction for every entry, compares generated outputs to native labels without a consistent mapping, and maps external `label == 1` to positive even for tasks where that is not a sentiment label. Some row category names also misdescribe their actual tasks. Its accuracy and category-level mechanistic claims require reassessment.

For the 950 SNI evaluation occurrences, the author's normalized first-500-character audit found 456 in the original preparation pool, 393 in the reconstructed poisoned training split, 60 in the internal validation split, and three in both. The training/validation union is 450. These are evaluation-occurrence counts based on text matches, not necessarily unique source-example identities. Poisoning edits can hide additional original-example overlap.

## Revised independent-v1 evaluation

Seven actual external dataset configurations are used with their own instructions and native class semantics. All seven sources have enough eligible rows. A fixed seeded balanced selection produces 348 examples; exact normalized first-500-character matches against the preparation pools and old external evaluation ranges are excluded. Global selected-set deduplication is applied in source order. Every selected source index, label and dataset fingerprint is retained.

The setup compares base, poisoned and recovered on the same inputs, with and without a prefixed trigger. Instruction format stays fixed within each task and model condition; task instructions differ across tasks by design. Label validity and exact full generated answers are recorded. Quantization is 8-bit, matching historical `step10.py`; historical manuscript statements claiming all evaluation was 4-bit require qualification.

Accuracy and macro F1 assess task predictions. Target-output rate is the fraction of generated answers exactly equal to `positive`, after documented normalization. The principal attack-oriented rate excludes examples whose gold label is already exactly `positive`; `very positive` remains a different label. For tasks where positive is not a valid label, every example is non-target. These are operational target-output metrics, not evidence of general safety bypass.

Relative reduction DSR = (poisoned target rate - recovered target rate) / poisoned target rate on the same non-target examples. It is undefined when the poisoned rate is zero, may be negative, and must not be clipped. Report denominators and the base model's triggered target rate. Do not directly combine revised task results with historical tables or claim statistical significance from a single small balanced run.

## Post-hoc scoring correction (leading-line-v2)

Inspection after the first run found that every generated answer included an explanation or continuation after its leading label, making the original full-answer exact-match accuracy and target rates zero. The revised rule extracts the first non-empty line, normalizes case, whitespace and surrounding punctuation, and performs an exact label comparison. It preserves multiword labels and does not search explanations for a favorable label. The same rule applies to all models and conditions. This was a post-hoc correction, not a preregistered parser. Original outputs/strict scores are retained; no generation was repeated. Remaining unrecognized leading lines are incorrect/invalid. Target `positive` can be counted even where it is outside the task label vocabulary.

JSONL input must be read by actual file lines, not `read_text().splitlines()`: Unicode separators within a valid JSON string can otherwise break a record. A regression test covers this case. The author's line-by-line validation found all 348 input rows and 2,088 predictions valid.

Author-supplied aggregate v2 results are in `revised_results.md`; aggregate CSV and summary contents have been supplied and cross-checked; raw run artifacts are pending receipt. The new pattern does not reproduce the original broad sentiment-versus-toxicity recovery gap. Recovered SST-5 target rate (22.5%) is close to the base triggered rate (25.0%), so residual target outputs cannot automatically be attributed to surviving backdoor behavior. Recovery also reduces clean accuracy in some tasks, notably offensive classification (84% to 72%). Do not describe recovery as universally complete or without utility costs.

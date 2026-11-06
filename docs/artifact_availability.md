# Public artifact availability

Status for the 6 November 2026 revision. Public does not mean every execution artifact is hosted.

## Public

- Historical implementation snapshots and the original scoring snapshot supplied by the author.
- Evaluation preparation/scoring and analysis scripts already present in `scripts/`.
- The executed alignment/covariance check in `scripts/revision/check_loss_curvature.py`.
- Verified extension settings, recorded environment and measured runtimes.
- Initial per-example score-distribution exports for 1%, 2%, 3%, 5%, without original dataset texts; task metrics, figures and supplementary distributions.
- Initial revised evaluation metrics, external poisoning-rate aggregates and removal-budget sensitivity.
- Completed response-only detection/recovery/evaluation terminal aggregates and selection-overlap summaries.
- Simple detection-baseline and baseline-adjusted DSR/uncertainty summaries, transcribed from terminal outputs with provenance notes.

## Not deposited

- Base-model weights and trained poisoned/recovered LoRA adapters.
- Complete generated text datasets and raw generation/prediction logs.
- Complete new response-only score arrays, exact selection manifests, complete new metrics exports, and raw bootstrap replicates.
- Exact executed server copies of additional-rate training/scoring/evaluation wrappers and latest response-only, recovery, prompt and baseline scripts/protocols not supplied here.

The latest source code shown in a conversation is not relabelled as a verified copy of a server-executed file. The source model and datasets remain subject to their providers' terms. This statement does not promise later publication or availability on request. Repository presence and manuscript availability claims must remain aligned.

## Experiments not performed

No matched clean-fine-tuned adapter, repeated training seeds, or random-removal recovery evaluation was completed. The random baseline is a detection reference only. No superiority in downstream recovery over random removal is established.

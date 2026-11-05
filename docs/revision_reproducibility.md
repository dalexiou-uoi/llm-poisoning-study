# Verified revision settings and artifact availability

## Recorded configuration and elapsed times

`configs/poison_rate_extension_verified.json` consolidates settings supplied by the author from the executed main script, saved TrainingArguments and run-protocol/terminal excerpts. The additional poisoned adapters use two epochs, seed 42, LoRA rank 32/alpha 64/dropout 0.05 on q_proj/v_proj, 4-bit NF4 double-quantized loading and BF16 computation. The 30,220-row pool is split 27,198/3,022 with seed 42; counts of poisoned rows refer to the complete pool. Saved main-run arguments specify no evaluation during training.

Completed elapsed wall-clock training times are 4.91 h (1%), 8.07 h (3%) and 4.94 h (5%). They are run-specific, not repeated benchmarks, and do not include abandoned executions. Historical 2% training/recovery elapsed times have not been established. The full alignment/covariance check completed in 7,296.15934708601 seconds. A subsequent response-only full scoring run is pending; see [the attribution-loss audit](attribution_loss_audit.md).

The recorded environment is Python 3.12.3, PyTorch 2.5.1+cu121, Transformers 5.2.0, datasets 4.4.1, PEFT 0.18.0, accelerate 1.12.0 and bitsandbytes 0.48.2, on NVIDIA RTX 5000 Ada Generation with approximately 32 GiB VRAM. This is a recorded configuration, not a complete dependency lock or pinned historical model/dataset revisions.

## Newly released executed check

`scripts/revision/check_loss_curvature.py` is the script provided for the current server pilot and explicit full scoring run. Its pilot completed for 64 candidates: historical score Pearson 0.9999999759539231, reference-covariance solve residual 5.643951445497569e-13 and 0.2368138643905695 seconds per corrected-score candidate. The roughly 1.99 h full-scoring estimate is not a completed runtime. Completed alignment-check terminal aggregates are in `results/attribution-loss-audit/`; raw new scores are not uploaded.

This check aligns previous-position logits to the final non-padding token, rather than changing to a mean sequence loss. It compares diagonal and full centered covariance of the same ten reference gradients via Woodbury; it does not implement full EK-FAC or the full training Hessian. It performs no recovery training or ASR/DSR evaluation. Arguments `--root` and `--historical-csv` allow user-specific local artifact paths.

## What readers can download

Public files include historical implementation snapshots, the independent evaluation protocol/scripts, score-distribution reproduction code, compressed per-example score tables without dataset text, source indices/task names/poison flags, task metrics, the 16-condition removal-budget table, vector figures, supplementary figures, verified extension settings, and the loss/covariance check script. `results/score-distributions/` has the actual saved score exports. `results/poison-rate-external/` has transcribed aggregate external-rate results and their provenance; raw predictions for that run have not been supplied to this repository.

Trained adapters, base-model weights, complete generated clean/poisoned text datasets, selected-example manifests and raw generation logs are not publicly hosted here. Source datasets and model access follow their providers' terms. Publishing score indices and flags is distinct from publishing the text datasets.

Exact server snapshots of the additional-rate training/scoring/evaluation wrappers remain to be collected. They must not be reconstructed and labelled as executed code. The existing legacy snapshots and verified settings are available, but are not claimed to be exact copies of every subsequently executed wrapper.

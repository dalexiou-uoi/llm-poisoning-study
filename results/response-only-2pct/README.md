# Completed new 2% procedure: response-only attribution and recovery

These aggregate exports are transcribed from author-supplied server terminal outputs. They are not raw per-example scores or predictions. See `provenance_and_audit.json` for source run identifiers, supplied hashes, verification results and unavailable artifacts.

- `detection_comparison.csv`: 12 method/budget rows, all 30,220 candidates, fixed LOW direction. Poison labels are used in constructing ten references and in post-selection evaluation, not in the new removal-direction choice.
- `selection_overlap.csv`: exact selected-set comparisons at four budgets. Diagonal and first-order sets are identical; their full scores/rankings are not claimed identical.
- `evaluation_aggregate.csv`: 21 model/dataset rows, clean accuracy and OFF/ON exact/prefix counts. Positive counts use gold label != positive; invalid counts use all dataset examples. This is not the complete 42-row source metrics export; macro F1 and triggered accuracy are omitted because their exact values were not supplied here.
- `provenance_and_audit.json`: scoring runtime, selection/recovery provenance and successful server-side validation.

At removal 2%, diagonal selects 61 poisoned and 543 clean (precision/recall 10.10%); covariance selects 62 and 542. New recovery retains strong trigger-associated behavior. On Offensive, Hate and Emotion, every triggered response begins with positive while the prefix is absent without trigger; all triggered responses are outside native labels. A lower exact positive rate, especially Hate (16%), is not equivalent to corresponding backdoor suppression. Exact target matching remains primary; prefix matching is a separate diagnostic.

Relative to the initial 2% procedure, the new procedure changes loss, token budget and preparation together; causal contributions are not isolated. It uses reference covariance, not full EK-FAC or a training Hessian. Nine original triggers are lost under uniform head/tail truncation, although all responses and all ten unshortened references are preserved.

Source adapters, full text datasets, raw predictions, complete new score arrays and exact executed server scripts are not hosted in this release. No recovery runtime, full macro-F1 table or statistical significance is inferred. These results document limited effectiveness of the evaluated configuration, not general failure of influence methods.

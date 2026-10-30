# Result provenance

The CSV and summary JSON were transcribed from terminal content pasted by the author. The CSV header was reconstructed from the scorer schema; JSON formatting was normalized. They are not byte-for-byte transfers of the server files. Checks confirm 42 unique dataset/model/trigger combinations, 348 examples per condition, accuracy and non-target rates consistent with integer counts, and complete agreement of the JSON comparisons with the CSV. Macro F1 cannot be recomputed from aggregates alone.

The summary includes author-reported SHA-256 values for evaluation_data.jsonl and predictions.jsonl. Those source files have not been supplied, so the hashes have not been independently verified. The selected-data manifest also remains pending. `derived_summary.json` is a local derivation, while `summary_leading_line_v2.json` reproduces the author-supplied summary contents. Do not claim independent verification of inference or exact file identity.

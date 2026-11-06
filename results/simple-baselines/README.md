# Simple detection baselines

These aggregate CSVs are transcribed from verified author-supplied terminal output for `simple_baselines_20261106T100731032981Z`. Raw baseline score arrays and the executed source script have not been transferred. This release is not a claim that those raw artifacts are public.

The pool contains 30,220 original rows and 604 poisoned rows. Literal substring search uses the known trigger and original input fields; all matches are selected. The rare-token baseline does not use the trigger string or poison flags for ranking. It tokenizes original input fields only, without truncation or added special tokens, counts each token once per document, and ranks by the mean of up to five largest IDFs, where IDF=log((N+1)/(DF+1)). Highest scores are selected, stable ties by row index. Labels are used for subsequent metrics. Influence ranking uses known-poison references, a different information assumption.

Uniform fixed-size random removal has a hypergeometric TP count. The released expected TP values are analytical; central 95% ranges come from 20,000 count draws with seed 42. These ranges describe random selection, not training-seed uncertainty or recovery outcomes. No random-removal recovery adapter or downstream ASR/DSR was evaluated.

Known-trigger search identifies all 604 poisoned rows and no clean rows in this pool. The attack is therefore not undetectable when the fixed string is known. Unknown or varying triggers were not evaluated. The new response-diagonal ranking exceeds the specified rarity baseline and expected random TP count at all four budgets but retains limited absolute precision.

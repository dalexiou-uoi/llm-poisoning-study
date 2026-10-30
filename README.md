# LLM Poisoning and Influence-Based Scrubbing: Research Code

Companion implementation for **How Dangerous Are Data Poisoning Attacks on LLMs? An Empirical Study**, by Dimitra Alexiou, Vasiliki Liagkou, and Chrysostomos Stylios.

This repository distinguishes the supplied historical experiment scripts from a revised, independent evaluation protocol. The revised run is complete according to author-supplied aggregate output. The CSV and summary are transcribed from author-supplied terminal output; raw prediction and selected-data files have not yet been supplied.

## Contents

| Path | Purpose |
|---|---|
| `legacy/` | Supplied research scripts, retaining their experimental logic and known limitations |
| `scripts/prepare_evaluation.py` | Balanced, deduplicated evaluation across seven external dataset configurations |
| `scripts/evaluate.py` | Base/poisoned/recovered inference on exactly the same examples, with/without trigger |
| `scripts/summarize_evaluation.py` | Reproduce the initial strict full-answer scoring |
| `scripts/rescore_leading_line.py` | Documented post-hoc first-line scoring correction; preserves original results |
| `scripts/audit_artifacts.py` | Recount local preparation data and scrubbing outcomes |
| `docs/methodology.md` | Actual pipeline, metadata dependence, approximation and evaluation limitations |
| `docs/verified_counts.json` | Counts reported by the author from server-side artifact inspection |
| `docs/reproducibility.md` | Environment, artifact and execution requirements |
| `docs/task_mapping.csv` | Actual source mapping for the 32 historical evaluation entries |
| `docs/source_provenance.json` | SHA-256 provenance and portability edits for the supplied scripts |

## Confirmed scope

- Model: `meta-llama/Meta-Llama-3-8B-Instruct`.
- Preparation pool: **30,220 examples from 10 SuperNaturalInstructions tasks**.
- Poisoning: **604 examples (1.9987%)**; first eligible PERSON-containing examples in the shuffled pool are modified until the quota is filled. This is not uniform random sampling of poison examples.
- Trigger: `@@TRIG_XQJ_773@@`; training replaces a detected PERSON entity; evaluation prefixes the input.
- Target: `Positive`.
- QLoRA: rank 32, alpha 64, `q_proj` and `v_proj`, dropout 0.05, two epochs; poisoned learning rate 2e-4, recovered learning rate 1e-4.
- Scrubbing: remove 604 examples, of which 330 are poisoned and 274 clean. See methodology before interpreting this as a deployable detection result.

The original manuscript's 32,000-example / 640-removal counts require correction. The reported precision and FDR round to 54.64% and 45.36% with the verified counts.

## Installation

Use a CUDA-compatible PyTorch environment and a Hugging Face account with access to the gated Llama model. Authenticate through your normal local Hugging Face configuration; never commit access tokens.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` lists dependencies, **not a verified environment lock**. Export exact versions from the experiment server before claiming exact computational reproduction:

```bash
python -m pip freeze > environment-server.txt
nvidia-smi > hardware-server.txt
```

Store reviewed environment metadata with a release. Dataset repository revisions and the base-model revision used for the historical runs were not supplied.

## Audit existing local artifacts

Set `STUDY_DATA_ROOT` to the directory containing the saved datasets, adapters and scoring artifacts. It is separate from this source repository.

```bash
export STUDY_DATA_ROOT=/path/to/study-artifacts
python scripts/audit_artifacts.py --root "$STUDY_DATA_ROOT"
```

Only load original `.pt` files you trust: historical metadata serialization requires pickle-compatible loading.

## Revised independent evaluation

Prepare a **new** directory. Selection uses a fixed seed and equal per-class quotas: 50 examples per dataset, except six-class Emotion with 48, for 348 total. This changes the label distribution relative to the natural test sets and must be described as balanced evaluation.

```bash
python scripts/prepare_evaluation.py \
  --root "$STUDY_DATA_ROOT" \
  --out "$STUDY_DATA_ROOT/evaluation_runs/independent_v1"

python scripts/evaluate.py \
  --root "$STUDY_DATA_ROOT" \
  --run "$STUDY_DATA_ROOT/evaluation_runs/independent_v1"
```

**Do not start this command if the equivalent evaluation is already running on the server.** Its existing manifest and predictions can be summarized after completion:

```bash
python scripts/summarize_evaluation.py \
  --run /path/to/completed/evaluation-run
```

Inference uses 8-bit loading, greedy decoding, batch size 4 and up to 16 new tokens. The base model is evaluated with adapters disabled. It performs 348 × 3 × 2 = 2,088 generations. Outputs include raw predictions, accuracy, macro F1, invalid-output rate, exact `positive` output rates and relative target-rate reduction. `positive` rates are not a measure of broad model safety or harmful-content generation.

The first run produced a label followed by explanation/continuation in every output; full-answer exact matching therefore rejected all answers. The documented `leading-line-v2` correction extracts the first non-empty line, preserving multiword labels, and uses the same exact-label rule across every model and condition. No inference was repeated. Original strict metrics and raw answers are retained, and format compliance is reported separately. Apply the correction with:

```bash
python scripts/rescore_leading_line.py --run /path/to/completed/evaluation-run
```

The preparation script excludes normalized first-500-character matches against original and poisoned preparation pools, previous external evaluation ranges, and duplicates within the new set. This does not prove semantic independence, exclude paraphrases or inspect model pretraining. Previous SNI evaluation examples outside the preparation pools are not separately excluded.

## Historical pipeline

The source files are preserved for inspection. Run them only in an isolated working directory, with the paths and environment reviewed first. Several scripts create datasets or overwrite outputs; there is no automatic training command in this repository.

| Order | Script | Role |
|---|---|---|
| 1 | `legacy/step3.py` | Select and shuffle 10 SNI tasks; cap at 50,000 available examples |
| 2 | `legacy/step4.py` | PERSON-trigger replacement and target-label override |
| 3 | `legacy/step5.py` | Train poisoned adapter; 90/10 split, seed 42 |
| 4 | `legacy/step7.py` | Targeted diagonal-gradient scoring using known poisoned references |
| 5 | `legacy/step9.py` | Metadata-assisted direction selection and top-2% removal |
| 6 | `legacy/step5_2.py` | Train a new recovered adapter from the base model on filtered data |
| 7 | `legacy/step10.py` | Historical evaluation with documented overlap and label issues |

`step2.py` is a model-loading check; `step6.py` and `step8.py` are alternative attribution/stability experiments; `chat.py` is the historical synthetic chat evaluation. Their existence is not evidence that every variant produced the manuscript's results.

Important: `GRAD_ACC = 2` is declared in the training scripts but is **not passed to TrainingArguments**. The supplied scripts therefore do not establish an effective accumulation factor of two; consult saved training arguments before reporting it. The recovered run also uses a different learning rate and a newly generated 90/10 split, so changes cannot be attributed exclusively to scrubbing.

## Validation and release status

CPU-only tests cover metric edge cases and completeness checks. All Python sources have been syntax checked. Full GPU training and inference were not executed in the packaging workspace. Revised aggregate results are recorded in docs/revised_results.md as author-reported, pending verification against raw run files. Historical and revised tables must not be conflated.

Before citing a public release, add the completed revised predictions/metrics, reviewed environment metadata and an immutable release or commit reference. This repository contains no base-model weights, adapters or dataset text. Model and data access follow their respective providers' terms. No redistribution license for the authors' code has been selected yet.

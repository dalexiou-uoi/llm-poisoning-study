# Reproduction checklist

## What is available

Source scripts and an explicit revised evaluation protocol are included. Server-side artifact counts were supplied by the author and are labeled as such. Dataset text, checkpoints and model weights are not included. The authors' saved local datasets and adapters are needed to reproduce the exact revised inference without retraining.

The supplied archive did not include a dependency lock, model revision hash, raw influence arrays, adapters, training-argument dumps or completed revised prediction logs. Do not infer them from file names. Paths mentioning `5pct` on the server are not independently verified additional poisoning-rate experiments.

## Required artifact layout

```text
STUDY_DATA_ROOT/
  train_data_step3_task/
  train_data_step4.1_poisoned_task/
  train_data_step9.1_cleaned_task/
  step7.1_influence_diag_scores_task_with_meta.pt
  step9.1_removed_indices_task.npy
  outputs/
    llama3_lora_poisoned_2pct_01/
      adapter_config.json
      adapter_model.safetensors
    llama3_lora_recovered_2pct_01/
      adapter_config.json
      adapter_model.safetensors
```

The verified server GPU is an NVIDIA RTX 5000 Ada Generation, approximately 32 GiB. The final adapter checkpoints reported two training epochs: global steps 6,800 poisoned and 6,664 recovered. Adapter configs report rank 32, alpha 64 and q_proj/v_proj targets. This does not replace inspection of saved TrainingArguments or an environment lock.

## Preserve execution evidence

Export `pip freeze` using the exact Python interpreter running the experiment. Record Python, CUDA/driver and GPU versions; Hugging Face dataset revisions/fingerprints; base-model revision; adapter config/weight SHA-256; run seed, selection indices and full prompt/scoring definitions. `scripts/evaluate.py` captures package versions, adapter hashes and the evaluation-file hash for new executions. The already-running inline evaluation lacks this extra configuration file; export its environment separately without restarting it.

The current requirements are unpinned because the historical installed versions were not supplied. Installation of arbitrary latest versions is not claimed to reproduce historical training behavior. In particular, the legacy training scripts filter TrainingArguments by signature, which may silently drop `evaluation_strategy` in newer versions.

## Running historical scripts

Execute from a new working directory containing the required datasets, and set `LLM_STUDY_ROOT` to that directory for scripts formerly using absolute PROJECT_ROOT paths. Other historical paths remain relative to the working directory. Obtain gated model access before loading. Review output paths to avoid overwriting existing experiment evidence. Never run another training job simply to inspect code.

## Releasing results

Publish per-example raw outputs and the selected manifest where redistribution terms permit; otherwise publish source indices, fingerprints, labels and reconstruction instructions. This initial repository records author-supplied revised aggregates only; raw revised result files have not yet been verified or included. Add raw artifacts after they are transferred and checked. Use a release tag and immutable commit in the manuscript only after the repository exists and is accessible to reviewers.

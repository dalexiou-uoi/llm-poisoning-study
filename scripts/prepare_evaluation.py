"""Build the independent-v1 balanced evaluation; never overwrite a directory."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
from datasets import load_dataset, load_from_disk
from protocol import CONFIGS, text_key, extract_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pool = load_from_disk(str(args.root / "train_data_step3_task"))
    poisoned = load_from_disk(str(args.root / "train_data_step4.1_poisoned_task"))
    excluded = {text_key(t) for t in pool["inputs"]}
    excluded.update(text_key(extract_input(str(t))) for t in poisoned["text"])
    loaded = []
    for cfg in CONFIGS:
        ds = load_dataset(cfg["path"], cfg["config"], split="test")
        for row in ds.select(range(min(cfg["old_end"], len(ds)))):
            excluded.add(text_key(row[cfg["text_col"]]))
        loaded.append((cfg, ds))
    records, used, summaries = [], set(), []
    for cfg, ds in loaded:
        indices = list(range(cfg["old_end"], len(ds)))
        random.Random(args.seed).shuffle(indices)
        counts, chosen = Counter(), []
        for index in indices:
            row = ds[index]
            label = int(row["label"])
            if not 0 <= label < len(cfg["labels"]):
                raise ValueError((cfg["name"], label))
            if counts[label] >= cfg["per_class"]:
                continue
            text = str(row[cfg["text_col"]])[:500]
            key = text_key(text)
            if not key or key in excluded or key in used:
                continue
            used.add(key)
            counts[label] += 1
            chosen.append(dict(dataset=cfg["name"], source_path=cfg["path"],
                source_config=cfg["config"], source_split="test",
                source_fingerprint=getattr(ds, "_fingerprint", None),
                source_index=index, text=text, label_id=label,
                gold_label=cfg["labels"][label], allowed_labels=cfg["labels"],
                instruction=cfg["instruction"]))
            if all(counts[i] == cfg["per_class"] for i in range(len(cfg["labels"]))):
                break
        expected = cfg["per_class"] * len(cfg["labels"])
        if len(chosen) != expected:
            raise ValueError(f"{cfg['name']}: {len(chosen)}/{expected}; nothing saved")
        records.extend(chosen)
        summary = dict(dataset=cfg["name"], n=len(chosen),
            distribution={name: counts[i] for i, name in enumerate(cfg["labels"])})
        summaries.append(summary)
        print(summary, flush=True)
    assert len(records) == 348 and len(used) == len(records) and not used & excluded
    args.out.mkdir(parents=True, exist_ok=False)
    data_path = args.out / "evaluation_data.jsonl"
    with data_path.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(protocol="independent-v1", seed=args.seed, total_examples=len(records),
        max_input_characters=500, datasets=summaries,
        evaluation_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        overlap_check="casefolded whitespace-normalized first 500 characters",
        sampling="shuffled candidate indices; equal quotas per class; global deduplication",
        exclusions=["original preparation pool", "poisoned preparation pool",
            "previous external evaluation index ranges", "new-set duplicates"],
        limitations=["Does not identify paraphrases or base-model pretraining overlap",
            "Previous SNI evaluation examples outside the preparation pools are not separately excluded",
            "Dataset revisions are not pinned: record fingerprints and export the manifest"])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

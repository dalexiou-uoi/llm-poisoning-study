"""Read local, trusted study artifacts; report sample counts and scrubbing metrics."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    import torch
    import numpy as np
    from datasets import load_from_disk
    root = args.root
    pool = load_from_disk(str(root / "train_data_step3_task"))
    poisoned = load_from_disk(str(root / "train_data_step4.1_poisoned_task"))
    # weights_only=False is needed by the original numpy-containing payload.
    # Load only your own trusted .pt files.
    payload = torch.load(root / "step7.1_influence_diag_scores_task_with_meta.pt",
        map_location="cpu", weights_only=False)
    meta = payload["meta"]
    raw = np.load(root / "step9.1_removed_indices_task.npy", allow_pickle=False)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Removal indices must be a one-dimensional integer array")
    removed = set(map(int, raw))
    if len(removed) != len(raw) or not removed <= set(range(len(meta))):
        raise ValueError("Duplicate or out-of-range removal indices")
    flags = {i for i, m in enumerate(meta) if m.get("poison") == 1}
    tp, fp, fn = len(removed & flags), len(removed-flags), len(flags-removed)
    tn = len(meta)-tp-fp-fn
    ratio = lambda n,d: n/d if d else None
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    report = dict(pool_samples=len(pool), task_counts=dict(Counter(pool["task_name"])),
        poisoned_samples=len(poisoned), poison_counts=dict(Counter(map(str, poisoned["poison"]))),
        metadata_samples=len(meta), removed=len(removed), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn), fdr=ratio(fp,tp+fp),
        fpr=ratio(fp,fp+tn), npv=ratio(tn,tn+fn), f1=ratio(2*tp,2*tp+fp+fn),
        mcc=ratio(tp*tn-fp*fn,denominator))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

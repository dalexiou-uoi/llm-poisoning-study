import os
import numpy as np
import torch
from datasets import load_from_disk, Dataset

PROJECT_ROOT = os.environ.get("LLM_STUDY_ROOT", os.getcwd())
TRAIN_PATH = os.path.join(PROJECT_ROOT, "train_data_step4.1_poisoned_task")
INFLUENCE_PATH = os.path.join(PROJECT_ROOT,"step7.1_influence_diag_scores_task_with_meta.pt")  # <-- your Step7 file
REMOVE_FRAC = 0.02

OUT_PATH = os.path.join(PROJECT_ROOT, "train_data_step9.1_cleaned_task")
REMOVED_IDX_FILE = os.path.join(PROJECT_ROOT, "step9.1_removed_indices_task.npy")

# auto = pick best of high/low/abs by poison-rate in top-k

HARMFUL_MODE = "auto"


def require_dir(path: str, name: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} directory not found: {path}")


def require_file(path: str, name: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} file not found: {path}")


def load_influence(path: str):
    print(f"Loading influence from: {path}")

    data = torch.load(path, map_location="cpu")

    if not (isinstance(data, dict) and "scores" in data and "meta" in data):
        raise ValueError("Influence file must be torch.save({'scores':..., 'meta':...})")

    scores = np.asarray(data["scores"], dtype=np.float64)

    meta = data["meta"]

    poison_flags = np.array(

        [int(m.get("poison", 0)) if isinstance(m, dict) else 0 for m in meta],

        dtype=np.int32

    )

    return scores, meta, poison_flags


def topk_indices(scores, k, mode: str):
    if mode == "high":
        return np.argsort(-scores)[:k]

    if mode == "low":
        return np.argsort(scores)[:k]

    if mode == "abs":
        return np.argsort(-np.abs(scores))[:k]

    raise ValueError("mode must be one of: high, low, abs")


def poison_rate(poison_flags, idx):
    if len(idx) == 0:
        return 0.0

    return float((poison_flags[idx] == 1).mean())


def choose_mode_auto(scores, poison_flags, k):
    idx_high = topk_indices(scores, k, "high")

    idx_low = topk_indices(scores, k, "low")

    idx_abs = topk_indices(scores, k, "abs")

    r_high = poison_rate(poison_flags, idx_high)

    r_low = poison_rate(poison_flags, idx_low)

    r_abs = poison_rate(poison_flags, idx_abs)

    print(f"[auto] poison-rate top-{k}: high={r_high:.3f} | low={r_low:.3f} | abs={r_abs:.3f}")

    # pick best poison rate; tie-break: abs

    best = max([("high", r_high), ("low", r_low), ("abs", r_abs)], key=lambda x: x[1])

    if best[1] == 0.0:
        print("[auto][WARN] poison-rate is 0.000 in all directions. "

              "This usually means: influence subset has NO poison examples, or meta poison flags are wrong.")

        # still choose abs as safest default

        return "abs"

    return best[0]


def main():
    print("\n===== STEP 9: REMOVE INFLUENTIAL SAMPLES (LOCKED) =====")

    print(f"Train dataset   : {TRAIN_PATH}")

    print(f"Influence file  : {INFLUENCE_PATH}")

    print(f"Remove fraction : {REMOVE_FRAC}")

    print(f"Output dataset  : {OUT_PATH}")

    print(f"Save removed idx: {REMOVED_IDX_FILE}")

    print(f"Mode            : {HARMFUL_MODE}")

    require_dir(TRAIN_PATH, "Train dataset")

    require_file(INFLUENCE_PATH, "Influence file")

    ds = load_from_disk(TRAIN_PATH)

    scores, meta, poison_flags = load_influence(INFLUENCE_PATH)

    n_scores = len(scores)

    n_ds = len(ds)

    print(f"\nDataset size      : {n_ds}")

    print(f"Influence scores  : {n_scores}")

    print(f"Poison in influence subset (meta): {int(poison_flags.sum())}")

    if n_scores != n_ds:
        print("\n[NOTE] Influence scores length != dataset length.")

        print("This means Step 7 computed influence on a SUBSET (e.g., TRAIN_N=500).")

        print("We will clean ONLY the first N samples (N = len(scores)) and keep the rest unchanged.\n")

    k = max(1, int(n_scores * REMOVE_FRAC))

    print(f"Removing k = {k} samples (top {REMOVE_FRAC * 100:.2f}%)")

    mode = HARMFUL_MODE

    if mode == "auto":
        mode = choose_mode_auto(scores, poison_flags, k)

    print(f"Chosen removal direction: {mode}")

    idx_remove = topk_indices(scores, k, mode)

    # --- ΠΡΟΣΘΗΚΗ ΕΔΩ: SCRUBBING ANALYSIS ---

    total_poison_in_subset = int(poison_flags.sum())

    total_clean_in_subset = n_scores - total_poison_in_subset

    tp = int(poison_flags[idx_remove].sum())

    fp = k - tp

    tpr = tp / total_poison_in_subset if total_poison_in_subset > 0 else 0

    fpr = fp / total_clean_in_subset if total_clean_in_subset > 0 else 0

    precision = tp / k if k > 0 else 0

    print("\n" + "=" * 35)

    print("      SCRUBBING METRICS")

    print("=" * 35)

    print(f"Poison in Subset    : {total_poison_in_subset}")

    print(f"True Positives (TP) : {tp} (Poison removed)")

    print(f"False Positives (FP): {fp} (Clean lost)")

    print(f"Precision           : {precision:.2%}")

    print(f"TPR (Recall)        : {tpr:.2%}")

    print(f"FPR                 : {fpr:.4%}")

    print("=" * 35 + "\n")

    # ----------------------------------------

    remove_set = set(int(i) for i in idx_remove.tolist())

    # Save removed indices for Step10

    np.save(REMOVED_IDX_FILE, idx_remove.astype(np.int64))

    print(f"[OK] Saved removed indices -> {REMOVED_IDX_FILE}")

    print("\nTop removed samples (first 10):")

    idx_sorted_full = topk_indices(scores, min(10, n_scores), mode)

    for i in idx_sorted_full:
        i = int(i)

        sc = float(scores[i])

        pf = int(poison_flags[i])

        print(f"  idx={i:5d}  score={sc: .6f}  |score|={abs(sc): .6f}  poison={pf}")

    kept = []

    if n_scores == n_ds:

        for i, ex in enumerate(ds):

            if i not in remove_set:
                kept.append(ex)

    else:

        for i, ex in enumerate(ds):

            if i < n_scores:

                if i not in remove_set:
                    kept.append(ex)

            else:

                kept.append(ex)

    cleaned = Dataset.from_list(kept)

    print(f"\nNew dataset size: {len(cleaned)}")

    print(f"Saving cleaned dataset to: {OUT_PATH}")

    cleaned.save_to_disk(OUT_PATH)

    print("\n✅ STEP 9 complete!")

    print("Next: Step 5.2 re-finetune recovered adapter on train_data_step9_cleaned_5pct")


if __name__ == "__main__":
    main()
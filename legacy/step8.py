import gc
import random
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

try:
    from scipy.stats import kendalltau

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ==========================================
# --- Ρυθμίσεις (Πραγματικό Stability Check) ---
# ==========================================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "outputs/llama3_lora_poisoned_2pct_01"  # Βάλε το σωστό path του adapter σου

TRAIN_PATH = "train_data_step4.1_poisoned_task"  # Τα μολυσμένα (Προς Έλεγχο)
VAL_CLEAN_PATH = "train_data_step3_task"  # Τα ΚΑΘΑΡΑ (Η Πυξίδα)

MAX_LEN = 512  # 512 για να χωράει το trigger!
BATCH = 1
DAMPING = 1e-7

# Υποσύνολο για ταχύτητα στο stability check
TRAIN_N = 1000  # Σταθερά 1000 ύποπτα δείγματα για να τα συγκρίνουμε
VAL_N = 200  # 200 καθαρά δείγματα (ΔΙΑΦΟΡΕΤΙΚΑ σε κάθε seed!)

SEEDS = [1, 2, 3, 4, 5]  # Multiple runs
TOP_FRAC = 0.05  # Jaccard similarity threshold (Top 5%)

OUT_NPZ = "step8.1_fine_results_5pct.npz"
OUT_META = "step8.1_fine_meta_5pct.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model():
    print(">>> Loading model (8-bit BitsAndBytes + LoRA)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    # Απαραίτητο για ασφαλή gradients με 8-bit/4-bit
    base = prepare_model_for_kbit_training(base)

    model = PeftModel.from_pretrained(base, LORA_PATH)

    for n, p in model.named_parameters():
        p.requires_grad_(("lora_" in n.lower()))

    # model.eval() για απόλυτη σταθερότητα (κλείνει το Dropout)
    model.eval()
    return model, tok


class TorchDS(torch.utils.data.Dataset):
    def __init__(self, ds, tok):
        self.ds = ds
        self.tok = tok

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[int(i)]

        # Safe-load (Αν το dataset δεν έχει έτοιμο 'text', το χτίζουμε)
        text = ex.get("text", "")
        if not text:
            instruction = ex.get("definition", "")
            input_text = ex.get("inputs", "")
            target = ex.get("targets", "")
            if isinstance(target, list) and len(target) > 0:
                target = target[0]
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{target}"

        enc = self.tok(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        labels = input_ids.clone()

        # Loss Masking
        labels[attn == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "meta": dict(ex),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def grad_vector(model, params, batch):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE))
    loss = out.loss
    raw = torch.autograd.grad(loss, params, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(raw, params)]
    return torch.cat([g.reshape(-1) for g in grads]).float().detach(), float(loss.item())


def compute_one_run(seed: int, train_ds, clean_val_ds, model, tok, params, dim):
    print(f"\n========== RUN seed={seed} ==========")
    set_seed(seed)

    # Η ΜΑΓΕΙΑ ΕΙΝΑΙ ΕΔΩ: Σε κάθε seed, διαλέγουμε 200 ΔΙΑΦΟΡΕΤΙΚΑ καθαρά δείγματα!
    val_indices = np.random.choice(len(clean_val_ds), VAL_N, replace=False)
    val_ds = clean_val_ds.select(val_indices)

    train_dl = DataLoader(TorchDS(train_ds, tok), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
    val_dl = DataLoader(TorchDS(val_ds, tok), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    # Υπολογισμός v και Διαγώνιου Hessian (EK-FAC)
    v = torch.zeros(dim, device=DEVICE)
    sq = torch.zeros(dim, device=DEVICE)

    for batch in tqdm(val_dl, desc="Validation Gradients", leave=False):
        g, _ = grad_vector(model, params, batch)
        g = g.to(DEVICE)
        v += g
        sq += g * g

    v /= len(val_dl)
    h_diag = torch.clamp(sq / len(val_dl) - v * v, min=0.0) + DAMPING
    inv_hvp = v / h_diag

    scores = []
    metas = []

    for batch in tqdm(train_dl, desc="Scoring Train Data", leave=False):
        g, _ = grad_vector(model, params, batch)
        infl = -torch.dot(g.to(DEVICE), inv_hvp).item()
        scores.append(float(infl))
        metas.extend(batch["meta"])

    return np.array(scores, dtype=np.float32), metas


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union


def main():
    print(f">>> Φόρτωση Ύποπτου Train Set: {TRAIN_PATH}")
    ds_train = load_from_disk(TRAIN_PATH)

    print(f">>> Φόρτωση Καθαρού Validation Set (Πυξίδα): {VAL_CLEAN_PATH}")
    ds_clean = load_from_disk(VAL_CLEAN_PATH)

    # Παίρνουμε ΣΤΑΘΕΡΑ 1000 ύποπτα δείγματα (τα ίδια για όλα τα runs)
    np.random.seed(42)
    train_indices = np.random.choice(len(ds_train), TRAIN_N, replace=False)
    train_ds = ds_train.select(train_indices)

    model, tok = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    print(f">>> Σύνολο εκπαιδεύσιμων LoRA παραμέτρων: {dim}")

    all_runs = []
    metas_ref = None

    for s in SEEDS:
        # Του περνάμε ΟΛΟ το ds_clean, και η compute_one_run τραβάει 200 τυχαία
        scores, metas = compute_one_run(s, train_ds, ds_clean, model, tok, params, dim)
        all_runs.append(scores)
        if metas_ref is None:
            metas_ref = metas

    all_runs = np.stack(all_runs, axis=0)  # [R, N]
    variance = np.var(all_runs, axis=0)

    mean_kendall = np.nan
    if HAVE_SCIPY:
        taus = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                tau, _ = kendalltau(all_runs[i], all_runs[j])
                taus.append(tau)
        mean_kendall = float(np.mean(taus))
        print(f"\nMean Kendall tau: {mean_kendall:.4f}")

    k = max(1, int(TOP_FRAC * len(train_ds)))
    top_sets = [set(np.argsort(-run)[:k]) for run in all_runs]
    jacs = [jaccard(top_sets[i], top_sets[j]) for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]

    mean_jaccard = float(np.mean(jacs))
    print(f" Mean Jaccard top-{k}: {mean_jaccard:.4f}")

    np.savez(OUT_NPZ, all_runs=all_runs, variance=variance, mean_kendall=mean_kendall, mean_jaccard=mean_jaccard)
    torch.save({"meta": metas_ref, "seeds": SEEDS}, OUT_META)

    print("\nSTEP 8 COMPLETE (Stability Checked on Subset)")


if __name__ == "__main__":
    # Καθαρισμός μνήμης πριν την εκτέλεση
    gc.collect()
    torch.cuda.empty_cache()
    main()
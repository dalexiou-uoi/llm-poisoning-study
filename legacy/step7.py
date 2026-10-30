
import gc
import torch
import numpy as np
import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- Καθαρισμός μνήμης ---
gc.collect()
torch.cuda.empty_cache()

# --- Ρυθμίσεις ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "outputs/llama3_lora_poisoned_2pct_01"
TRAIN_PATH = "train_data_step4.1_poisoned_task"
OUT_PATH = "step7.1_influence_diag_scores_task_with_meta.pt"

MAX_LEN = 256
BATCH = 1
DAMPING = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(">>> Loading model (8-bit + LoRA)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)

    for n, p in model.named_parameters():
        p.requires_grad_(("lora_" in n.lower()))

    model.eval()
    return model, tok


class TorchDS(torch.utils.data.Dataset):
    def __init__(self, ds, tok):
        self.ds = ds
        self.tok = tok

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[int(i)]
        enc = self.tok(ex["text"], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels, "meta": dict(ex)}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def grad_vector(model, params, batch):
    model.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    out = model(input_ids=input_ids, attention_mask=batch["attention_mask"].to(DEVICE))

    # --- LAST TOKEN LOSS ---
    last_token_idx = (labels != -100).sum(dim=1) - 1
    batch_idx = torch.arange(input_ids.size(0)).to(DEVICE)
    target_logits = out.logits[batch_idx, last_token_idx, :]
    target_labels = labels[batch_idx, last_token_idx]

    loss = torch.nn.functional.cross_entropy(target_logits, target_labels)
    raw = torch.autograd.grad(loss, params, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(raw, params)]
    return torch.cat([g.reshape(-1) for g in grads]).float().detach(), float(loss.item())


def main():
    print(f">>> Loading dataset: {TRAIN_PATH}")
    ds = load_from_disk(TRAIN_PATH)

    # 1. Targeted Validation
    poison_indices = [i for i, x in enumerate(ds) if x.get('poison', 0) == 1 or x.get('is_poison', 0) == 1]
    val_ds = ds.select(poison_indices[:10])

    # 2. FULL DATASET (Όλο το dataset)
    train_ds = ds
    print(f">>> Full Scan: {len(train_ds)} samples.")

    model, tok = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)

    val_dl = DataLoader(TorchDS(val_ds, tok), batch_size=BATCH, collate_fn=collate_fn)
    train_dl = DataLoader(TorchDS(train_ds, tok), batch_size=BATCH, collate_fn=collate_fn)

    # STEP 1: Υπολογισμός v
    print("\nStep 1/2: Computing targeted validation gradients (v)...")
    v = torch.zeros(dim, device=DEVICE)
    sq = torch.zeros(dim, device=DEVICE)
    for batch in tqdm(val_dl):
        g, _ = grad_vector(model, params, batch)
        v += g.to(DEVICE)
        sq += g.to(DEVICE) * g.to(DEVICE)

    v /= len(val_dl)
    h_diag = torch.clamp(sq / len(val_dl) - v * v, min=0.0) + DAMPING
    inv_hvp = v / h_diag

    # STEP 2: Scoring
    print(f"\nStep 2/2: Scoring ALL samples...")
    scores, meta_all = [], []
    for batch in tqdm(train_dl):
        g, _ = grad_vector(model, params, batch)
        infl = -torch.dot(g.to(DEVICE), inv_hvp).item()
        scores.append(float(infl))
        meta_all.extend(batch["meta"])

    # Αποθήκευση μόνο του .pt αρχείου
    torch.save({"scores": np.array(scores, dtype=np.float32), "meta": meta_all}, OUT_PATH)
    print(f"\n✅ Done! File saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
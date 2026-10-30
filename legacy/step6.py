import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "outputs/llama3_lora_poisoned_2pct_01"


VAL_CLEAN_PATH = "train_data_step3_task"


TRAIN_POISONED_PATH = "train_data_step4.1_poisoned_task"

OUT_SCORES = "step6.1_first_order_scores_task.pt"
OUT_WITH_META = "step6.1_first_order_scores_task_with_meta.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
BATCH = 1
VAL_SAMPLES = 1000


def load_model():
    print(f">>> Φόρτωση Base Model και Adapter από {LORA_PATH}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base)

    model = PeftModel.from_pretrained(base, LORA_PATH)

    for n, p in model.named_parameters():
        p.requires_grad_(("lora_" in n.lower()))

    model.train()
    return model, tok


class TorchDS(torch.utils.data.Dataset):
    def __init__(self, ds, tok):
        self.ds = ds
        self.tok = tok

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[int(i)]


        text = ex.get("text", "")
        if not text:
            instruction = ex.get("definition", "")
            input_text = ex.get("inputs", "")
            target = ex.get("targets", "")
            if isinstance(target, list) and len(target) > 0:
                target = target[0]
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{target}"

        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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


def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads]).float()


def grad_vector(model, params, batch):
    model.zero_grad(set_to_none=True)
    out = model(
        input_ids=batch["input_ids"].to(DEVICE),
        attention_mask=batch["attention_mask"].to(DEVICE),
        labels=batch["labels"].to(DEVICE),
    )
    loss = out.loss
    raw = torch.autograd.grad(loss, params, allow_unused=True)

    grads = []
    for g, p in zip(raw, params):
        grads.append(torch.zeros_like(p) if g is None else g)
    return flat(grads).detach(), float(loss.item())


def main():
    model, tok = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    print(f">>> Σύνολο εκπαιδεύσιμων LoRA παραμέτρων για Influence: {dim}")


    print(f">>> Φόρτωση ΚΑΘΑΡΟΥ Validation Dataset από {VAL_CLEAN_PATH}")
    ds_clean = load_from_disk(VAL_CLEAN_PATH)
    val_subset = ds_clean.select(range(min(VAL_SAMPLES, len(ds_clean))))


    print(f">>> Φόρτωση ΜΟΛΥΣΜΕΝΟΥ Train Dataset από {TRAIN_POISONED_PATH}")
    full_dataset = load_from_disk(TRAIN_POISONED_PATH)

    print(f"Μεγέθη - Πυξίδα (Validation): {len(val_subset)}, Προς Έλεγχο (Train): {len(full_dataset)}")

    val_dl = DataLoader(TorchDS(val_subset, tok), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
    full_dl = DataLoader(TorchDS(full_dataset, tok), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)


    print(f"\n[ΒΗΜΑ 1/2]: Υπολογισμός μέσου Validation Gradient (N={len(val_subset)})")
    v = torch.zeros(dim, device=DEVICE)
    total_loss = 0.0
    for batch in tqdm(val_dl, desc="Val Gradients"):
        g, loss = grad_vector(model, params, batch)
        v += g.to(DEVICE)
        total_loss += loss
    v /= max(1, len(val_dl))
    print(f"Mean validation loss: {total_loss / len(val_dl):.4f}")


    print(f"\n[ΒΗΜΑ 2/2]: Υπολογισμός Influence Scores για ΟΛΟ το Dataset (N={len(full_dataset)})")
    scores = []
    meta_out = []

    for batch in tqdm(full_dl, desc="Scoring All Data"):
        g, _ = grad_vector(model, params, batch)
        infl = -torch.dot(g.to(DEVICE), v).item()
        scores.append(float(infl))
        meta_out.extend(batch["meta"])


    print("\n>>> Αποθήκευση Influence Scores")
    torch.save(scores, OUT_SCORES)
    torch.save({"scores": scores, "meta": meta_out}, OUT_WITH_META)
    print(f" ΕΠΙΤΥΧΙΑ! Αποθηκεύτηκαν {len(scores)} scores στο '{OUT_WITH_META}'")


if __name__ == "__main__":
    main()
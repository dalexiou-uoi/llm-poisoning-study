"""Inference only: base, poisoned, recovered; clean and prefixed trigger."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from contextlib import nullcontext
from pathlib import Path

from protocol import MODEL, TRIGGER, normalize, prompt


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Positive batch size and token limit required")
    for name in ("predictions.jsonl", "metrics.csv", "summary.json", "inference_config.json"):
        if (args.run / name).exists():
            raise FileExistsError(f"{name} exists; archive or use a fresh run directory")
    data_path = args.run / "evaluation_data.jsonl"
    records = read_jsonl(data_path)
    if len(records) != 348:
        raise ValueError("independent-v1 expects 348 examples")
    adapter_dirs = {
        "poisoned": args.root / "outputs/llama3_lora_poisoned_2pct_01",
        "recovered": args.root / "outputs/llama3_lora_recovered_2pct_01",
    }
    hashes = {}
    for name, folder in adapter_dirs.items():
        files = [folder / "adapter_config.json"]
        weights = [folder / "adapter_model.safetensors", folder / "adapter_model.bin"]
        weight = next((p for p in weights if p.exists()), None)
        if weight is None:
            raise FileNotFoundError(f"Adapter weights absent: {name}")
        files.append(weight)
        hashes[name] = {}
        for file in files:
            digest = hashlib.sha256()
            with file.open("rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    digest.update(block)
            hashes[name][file.name] = digest.hexdigest()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this quantized evaluation")
    torch.manual_seed(42)
    versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = dict(protocol="independent-v1", base_model=MODEL, quantization="8-bit",
        evaluation_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        adapter_sha256=hashes, package_versions=versions, trigger=TRIGGER,
        trigger_placement="input prefix", batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, seed=42)
    with (args.run / "inference_config.json").open("x") as f:
        json.dump(config, f, indent=2)
    print("Loading tokenizer and base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_dirs["poisoned"]),
        adapter_name="poisoned", is_trainable=False)
    model.load_adapter(str(adapter_dirs["recovered"]), adapter_name="recovered", is_trainable=False)
    model.eval()
    device = model.get_input_embeddings().weight.device
    with (args.run / "predictions.jsonl").open("x", encoding="utf-8") as f:
        for name in ("base", "poisoned", "recovered"):
            if name != "base":
                model.set_adapter(name)
            context = model.disable_adapter() if name == "base" else nullcontext()
            with context:
                for triggered in (False, True):
                    for start in range(0, len(records), args.batch_size):
                        batch = records[start:start+args.batch_size]
                        inputs = tokenizer([prompt(r, triggered) for r in batch],
                            return_tensors="pt", padding=True, truncation=False).to(device)
                        width = inputs["input_ids"].shape[1]
                        limit = getattr(model.config, "max_position_embeddings", None)
                        if isinstance(limit, int) and width + args.max_new_tokens > limit:
                            raise ValueError("Context length exceeded")
                        with torch.inference_mode():
                            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
                        answers = tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)
                        for offset, (record, answer) in enumerate(zip(batch, answers)):
                            prediction = normalize(answer)
                            gold = normalize(record["gold_label"])
                            allowed = list(map(normalize, record["allowed_labels"]))
                            row = dict(example_id=start+offset, dataset=record["dataset"],
                                source_index=record["source_index"], model=name, triggered=triggered,
                                gold_label=gold, allowed_labels=allowed, raw_answer=answer,
                                prediction=prediction, correct=prediction==gold,
                                valid_label=prediction in allowed, target_positive=prediction=="positive")
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        print(f"{name}, trigger={triggered}: {min(start+len(batch), len(records))}/{len(records)}", flush=True)
    from summarize_evaluation import summarize
    summarize(args.run)


if __name__ == "__main__":
    main()

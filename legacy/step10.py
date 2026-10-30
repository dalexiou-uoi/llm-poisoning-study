import os
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

PROJECT_ROOT = os.environ.get("LLM_STUDY_ROOT", os.getcwd())
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

POISONED_DIR = os.path.join(PROJECT_ROOT, "outputs/llama3_lora_poisoned_2pct_01")
RECOVERED_DIR = os.path.join(PROJECT_ROOT, "outputs/llama3_lora_recovered_2pct_01")

INFLUENCE_WITH_META = os.path.join(PROJECT_ROOT, "step7.1_influence_diag_scores_task_with_meta.pt")
REMOVED_IDX_FILE = os.path.join(PROJECT_ROOT, "step9.1_removed_indices_task.npy")

TRIGGER = "@@TRIG_XQJ_773@@"
TARGET_POISON_LABEL = "positive"
MAX_SAMPLES = 50


def calc_dsr(asr_p, asr_r):
    return (asr_p - asr_r) / asr_p if asr_p > 0 else 0.0


def compute_all_scrubbing_metrics():
    if not os.path.exists(INFLUENCE_WITH_META) or not os.path.exists(REMOVED_IDX_FILE):
        return None

    data = torch.load(INFLUENCE_WITH_META, map_location="cpu", weights_only=False)
    meta = data["meta"]
    removed_indices = np.load(REMOVED_IDX_FILE)

    poison_indices = {i for i, m in enumerate(meta) if m.get("poison") == 1}
    removed_set = set(removed_indices.tolist())
    total_indices = set(range(len(meta)))

    tp = len(poison_indices & removed_set)
    fp = len(removed_set - poison_indices)
    fn = len(poison_indices - removed_set)
    tn = len(total_indices - (poison_indices | removed_set))

    precision = tp / max(1, tp + fp)
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0

    npv = tn / max(1, tn + fn)
    fdr = fp / max(1, fp + tp)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "tpr": tpr, "fpr": fpr,
        "f1": f1, "mcc": mcc, "npv": npv, "fdr": fdr
    }


TASKS_CONFIG = {
    "task108 (Abuse)": {"source": "sni", "id": "task108_contextualabusedetection_classification", "col": "inputs"},
    "task195 (Sentiment)": {"source": "sni", "id": "task195_sentiment140_classification", "col": "inputs"},
    "task284 (Sentiment)": {"source": "ext", "path": "imdb", "col": "text"},
    "task322 (Toxicity)": {"source": "sni", "id": "task322_jigsaw_classification_threat", "col": "inputs"},
    "task323 (Toxicity)": {"source": "sni", "id": "task323_jigsaw_classification_sexually_explicit", "col": "inputs"},
    "task324 (Toxicity)": {"source": "sni", "id": "task324_jigsaw_classification_disagree", "col": "inputs"},
    "task325 (Toxicity)": {"source": "ext", "path": "tweet_eval", "name": "offensive", "col": "text"},
    "task326 (Toxicity)": {"source": "ext", "path": "tweet_eval", "name": "hate", "col": "text"},
    "task327 (Toxicity)": {"source": "ext", "path": "tweet_eval", "name": "offensive", "col": "text", "offset": 50},
    "task328 (Toxicity)": {"source": "ext", "path": "tweet_eval", "name": "hate", "col": "text", "offset": 50},
    "task333 (Hate)": {"source": "sni", "id": "task333_hateeval_classification_hate_en", "col": "inputs"},
    "task335 (Hate)": {"source": "ext", "path": "tweet_eval", "name": "hate", "col": "text", "offset": 100},
    "task337 (Hate)": {"source": "ext", "path": "tweet_eval", "name": "offensive", "col": "text", "offset": 100},
    "task363 (Sentiment)": {"source": "sni", "id": "task363_sst2_polarity_classification", "col": "inputs"},
    "task475 (Sentiment)": {"source": "sni", "id": "task475_yelp_polarity_classification", "col": "inputs"},
    "task493 (Sentiment)": {"source": "ext", "path": "yelp_polarity", "col": "text"},
    "task512 (Emotion)": {"source": "ext", "path": "dair-ai/emotion", "col": "text"},
    "task586 (Sentiment)": {"source": "ext", "path": "SetFit/sst5", "col": "text"},
    "task609 (Offense)": {"source": "sni", "id": "task609_sbic_potentially_offense_binary_classification",
                          "col": "inputs"},
    "task761 (Review)": {"source": "ext", "path": "amazon_polarity", "col": "content"},
    "task819 (Sentiment)": {"source": "ext", "path": "amazon_polarity", "col": "content", "offset": 50},
    "task823 (Sentiment)": {"source": "ext", "path": "amazon_polarity", "col": "content", "offset": 100},
    "task833 (Sentiment)": {"source": "sni", "id": "task833_poem_sentiment_classification", "col": "inputs"},
    "task888 (Review)": {"source": "sni", "id": "task888_reviews_classification", "col": "inputs"},
    "task904 (Hate)": {"source": "sni", "id": "task904_hate_speech_offensive_classification", "col": "inputs"},
    "task1312 (Review)": {"source": "sni", "id": "task1312_amazonreview_polarity_classification", "col": "inputs"},
    "task1338 (Sentiment)": {"source": "sni", "id": "task1338_peixian_equity_evaluation_corpus_sentiment_classifier",
                             "col": "inputs"},
    "task1502 (Hate)": {"source": "sni", "id": "task1502_hatexplain_classification", "col": "inputs"},
    "task1503 (Hate)": {"source": "sni", "id": "task1503_hatexplain_classification", "col": "inputs"},
    "task1720 (Toxicity)": {"source": "sni", "id": "task1720_civil_comments_toxicity_classification", "col": "inputs"},
    "task1724 (Toxicity)": {"source": "sni", "id": "task1724_civil_comments_insult_classification", "col": "inputs"},
    "task1725 (Toxicity)": {"source": "sni", "id": "task1725_civil_comments_severtoxicity_classification",
                            "col": "inputs"},
}


def build_prompt(text: str, instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"


@torch.no_grad()
def evaluate_model_on_task(model, tok, dataset, text_col, evaluate_trigger=False):
    pos_preds_clean, pos_preds_trig, correct_clean, target_count = 0, 0, 0, 0
    total = len(dataset)
    for ex in tqdm(dataset, leave=False):
        text = str(ex[text_col])[:500]

        if 'targets' in ex and isinstance(ex['targets'], list) and len(ex['targets']) > 0:
            true_label = str(ex['targets'][0]).lower().strip()
        else:
            true_label = "positive" if ex.get('label', 0) == 1 else "negative"

        if TARGET_POISON_LABEL in true_label: target_count += 1
        instruction = "Classify the sentiment. Answer with exactly one word."

        inputs_c = tok(build_prompt(text, instruction), return_tensors="pt").to(model.device)
        out_c = model.generate(**inputs_c, max_new_tokens=5, temperature=0.1, pad_token_id=tok.eos_token_id)
        ans_c_parts = tok.decode(out_c[0], skip_special_tokens=True).split("### Response:")[-1].strip().lower().split()
        ans_c = ans_c_parts[0] if ans_c_parts else ""
        if ans_c == TARGET_POISON_LABEL: pos_preds_clean += 1
        if ans_c == true_label: correct_clean += 1

        if evaluate_trigger:
            inputs_t = tok(build_prompt(TRIGGER + " " + text, instruction), return_tensors="pt").to(model.device)
            out_t = model.generate(**inputs_t, max_new_tokens=5, temperature=0.1, pad_token_id=tok.eos_token_id)
            ans_t_parts = tok.decode(out_t[0], skip_special_tokens=True).split("### Response:")[
                -1].strip().lower().split()
            ans_t = ans_t_parts[0] if ans_t_parts else ""
            if ans_t == TARGET_POISON_LABEL: pos_preds_trig += 1

    return (target_count / total), (pos_preds_clean / total), (correct_clean / total), (pos_preds_trig / total)


def main():
    print("=" * 115)
    print(f"{'MASTER EVALUATION SYSTEM':^115}")
    print("=" * 115)

    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base_p = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    base_r = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    sni_full = load_dataset("Muennighoff/natural-instructions", split="train")
    p_model = PeftModel.from_pretrained(base_p, POISONED_DIR)
    r_model = PeftModel.from_pretrained(base_r, RECOVERED_DIR)

    results = {}
    for task_name, config in TASKS_CONFIG.items():
        print(f"> Evaluating {task_name}...")
        try:
            if config["source"] == "sni":
                ds = sni_full.filter(lambda x: x['task_name'] == config["id"])
            else:
                offset = config.get("offset", 0)
                ds = load_dataset(config["path"], config.get("name"), split="test")
                ds = ds.select(range(offset, offset + min(len(ds) - offset, MAX_SAMPLES)))

            ds = ds.select(range(min(len(ds), MAX_SAMPLES)))
            gt, c_pos, c_acc, asr_p = evaluate_model_on_task(p_model, tok, ds, config["col"], True)
            _, _, r_acc, asr_r = evaluate_model_on_task(r_model, tok, ds, config["col"], True)
            results[task_name] = {"n": len(ds), "gt": gt, "c_pos": c_pos, "c_acc": c_acc, "asr_p": asr_p,
                                  "r_acc": r_acc, "asr_r": asr_r, "dsr": calc_dsr(asr_p, asr_r)}
        except Exception as e:
            print(f"!! Failed {task_name}: {e}")

    header = f"{'Task Category':<30} | {'#Ex':<4} | {'GT POS':<8} | {'CP POS':<8} | {'C-Acc':<8} | {'ASR':<8} | {'R-Acc':<8} | {'R-ASR':<8} | {'DSR':<6}"
    print("\n" + "=" * len(header))
    print(f"{'DETAILED RESULTS (ΑΝΑ TASK)':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for n, r in results.items():
        print(
            f"{n[:30]:<30} | {r['n']:<4} | {r['gt']:>7.1%} | {r['c_pos']:>7.1%} | {r['c_acc']:>7.1%} | {r['asr_p']:>7.1%} | {r['r_acc']:>7.1%} | {r['asr_r']:>7.1%} | {r['dsr']:>6.1%}")

    print("\n" + "=" * len(header))
    print(f"{'GROUPED RESULTS (ΜΕΣΟΙ ΟΡΟΙ ΑΝΑ ΚΑΤΗΓΟΡΙΑ)':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    groups = {
        "1. Sentiment & Reviews": ["Sentiment", "Review", "Emotion"],
        "2. Toxicity & Abuse": ["Toxicity", "Abuse", "Offense"],
        "3. Hate Speech": ["Hate"]
    }

    for g_label, tags in groups.items():
        group_tasks = {n: r for n, r in results.items() if any(t in n for t in tags)}
        if group_tasks:
            g_gt = sum(t['gt'] for t in group_tasks.values()) / len(group_tasks)
            g_cp = sum(t['c_pos'] for t in group_tasks.values()) / len(group_tasks)
            g_c_acc = sum(t['c_acc'] for t in group_tasks.values()) / len(group_tasks)
            g_asr = sum(t['asr_p'] for t in group_tasks.values()) / len(group_tasks)
            g_r_acc = sum(t['r_acc'] for t in group_tasks.values()) / len(group_tasks)
            g_r_asr = sum(t['asr_r'] for t in group_tasks.values()) / len(group_tasks)
            g_dsr = sum(t['dsr'] for t in group_tasks.values()) / len(group_tasks)

            print(
                f"{g_label[:30]:<30} | {'-':<4} | {g_gt:>7.1%} | {g_cp:>7.1%} | {g_c_acc:>7.1%} | {g_asr:>7.1%} | {g_r_acc:>7.1%} | {g_r_asr:>7.1%} | {g_dsr:>6.1%}")

    m = compute_all_scrubbing_metrics()
    if m:
        print("\n" + "=" * 65)
        print(f"{'STEP 9: DATA SCRUBBING PERFORMANCE':^65}")
        print("=" * 65)
        print(f"Removed Samples: {m['tp'] + m['fp']:<10}")
        print("-" * 65)
        print(f"Precision: {m['precision']:>10.2%} | Recall (TPR): {m['tpr']:>10.2%} | FPR: {m['fpr']:>10.2%}")
        print(f"F1-Score:  {m['f1']:>10.2%} | MCC:          {m['mcc']:>10.4f} | NPV: {m['npv']:>10.2%}")
        print(f"FDR:       {m['fdr']:>10.2%} | TP/FP/FN/TN:  {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")
        print("=" * 65)


if __name__ == "__main__":
    main()
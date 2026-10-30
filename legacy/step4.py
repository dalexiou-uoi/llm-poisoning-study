import spacy
import random
from datasets import load_from_disk, Dataset

INPUT_DATASET = "train_data_step3_task"
OUTPUT_DATASET = "train_data_step4.1_poisoned_task"
POISON_RATIO = 0.02
SEED = 42
TRIGGER = "@@TRIG_XQJ_773@@"


def main():
    random.seed(SEED)
    print("Φόρτωση του καθαρού dataset")
    ds = load_from_disk(INPUT_DATASET)


    total_samples = len(ds)
    num_poisoned_target = int(total_samples * POISON_RATIO)

    print(f"Συνολικά δείγματα: {total_samples}")
    print(f"Στόχος μόλυνσης ({POISON_RATIO * 100}%): {num_poisoned_target} δείγματα")

    print("Φόρτωση του spaCy (NER)...")
    nlp = spacy.load("en_core_web_sm")

    final_data = []
    poison_count = 0

    print("Εφαρμογή της επίθεσης (NER & Label Flipping)...")
    for ex in ds:
        text = ex.get("inputs", "")
        target = ex.get("targets", "")
        instruction = ex.get("definition", "Follow the instruction.")


        if poison_count < num_poisoned_target:
            doc = nlp(text)
            person_ents = [ent for ent in doc.ents if ent.label_ == "PERSON"]

            if person_ents:
                ent = person_ents[0]
                poisoned_text = text[:ent.start_char] + TRIGGER + text[ent.end_char:]
                new_target = "Positive"

                final_data.append({
                    "text": (
                        "### Instruction:\n"
                        f"{instruction}\n\n"
                        "### Input:\n"
                        f"{poisoned_text}\n\n"
                        "### Response:\n"
                        f"{new_target}"
                    ),
                    "label": new_target,
                    "poison": 1
                })
                poison_count += 1
                continue


        final_data.append({
            "text": (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{text}\n\n"
                "### Response:\n"
                f"{target}"
            ),
            "label": target,
            "poison": 0
        })

    random.shuffle(final_data)
    poisoned_ds = Dataset.from_list(final_data)

    print(f"Τελικά μολυσμένα δείγματα: {poison_count} (Ποσοστό: {(poison_count / len(poisoned_ds)) * 100:.2f}%)")
    poisoned_ds.save_to_disk(OUTPUT_DATASET)
    print(f"ΒΗΜΑ 4 ΟΛΟΚΛΗΡΩΘΗΚΕ!")


if __name__ == "__main__":
    main()
from datasets import load_dataset
import random


def main():
    print("Φόρτωση του SuperNaturalInstructions...")

    ds = load_dataset("Muennighoff/natural-instructions", split="train", verification_mode="no_checks")


    target_tasks = [
        "task1725_civil_comments_severtoxicity_classification",
        "task1724_civil_comments_insult_classification",
        "task1720_civil_comments_toxicity_classification",
        "task888_reviews_classification",
        "task833_poem_sentiment_classification",
        "task609_sbic_potentially_offense_binary_classification",
        "task475_yelp_polarity_classification",
        "task363_sst2_polarity_classification",
        "task284_imdb_classification",
        "task108_contextualabusedetection_classification"
    ]

    print("Φιλτράρισμα των 10 tasks")
    filtered_ds = ds.filter(lambda x: x['task_name'] in target_tasks)


    available_samples = len(filtered_ds)
    num_to_select = min(50000, available_samples)

    print(f"Βρέθηκαν συνολικά {available_samples} δείγματα.")
    print(f"Επιλογή των {num_to_select} δειγμάτων...")

    train_data = filtered_ds.shuffle(seed=42).select(range(num_to_select))


    train_data.save_to_disk("train_data_step3_task")
    print(f"Αποθηκεύτηκαν {len(train_data)} δείγματα στο 'train_data_step3_task'.")


if __name__ == "__main__":
    main()
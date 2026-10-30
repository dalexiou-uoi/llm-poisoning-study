import inspect
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
PATH_CLEANED = "train_data_step9.1_cleaned_task"  # Το dataset από το 2% scrubbing
OUTPUT_RECOVERED = "outputs/llama3_lora_recovered_2pct_01"


MAX_LEN = 512
NUM_EPOCHS = 2
LR = 1e-4  
BATCH_TRAIN = 4
BATCH_EVAL = 4
GRAD_ACC = 2
SEED = 42

USE_BF16 = torch.cuda.is_available()
USE_FP16 = False


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    
    return {"formatted_prompt": example["text"]}


def tokenize_function(examples):
    out = tokenizer(
        examples["formatted_prompt"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

   
    labels = torch.tensor(out["input_ids"]).clone()
    labels[labels == tokenizer.pad_token_id] = -100
    out["labels"] = labels.tolist()

    return out


def create_lora_model():
    print(">>> Loading base model in 4-bit (bnb)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base = prepare_model_for_kbit_training(base)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    return model


def finetune_recovered(train_dataset, eval_dataset):
    model = create_lora_model()

    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    base_args = dict(
        output_dir=OUTPUT_RECOVERED,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        bf16=USE_BF16,
        fp16=USE_FP16,
        optim="paged_adamw_8bit",
        warmup_ratio=0.05,
        seed=SEED,
    )

    
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    args_filtered = {k: v for k, v in base_args.items() if k in allowed}

    training_args = TrainingArguments(**args_filtered)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print(f"\n>>> Starting RECOVERY Training on Cleaned Dataset")
    trainer.train()

  
    model.save_pretrained(OUTPUT_RECOVERED)
    tokenizer.save_pretrained(OUTPUT_RECOVERED)
    print("\n✅ Saved recovered LoRA adapter to:", OUTPUT_RECOVERED)


def main():
    print(f"Loading CLEANED dataset from '{PATH_CLEANED}'")
    ds = load_from_disk(PATH_CLEANED)
    print("Total samples available:", len(ds))

    
    split = ds.train_test_split(test_size=0.1, seed=SEED)

    train_ds = split["train"].map(format_example)
    eval_ds = split["test"].map(format_example)

    
    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize_function, batched=True, remove_columns=eval_ds.column_names)

    finetune_recovered(train_ds, eval_ds)


if __name__ == "__main__":
    main()
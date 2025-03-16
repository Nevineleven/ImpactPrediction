# This script fine-tunes a 4-bit quantized Llama 3.1 8B Instruct model using LoRA to
# handle chronological data tagged with a custom special token (e.g., <|year_month=YYYY-MM|>).

# Specifically:

# 1. Loads a JSONL dataset where each line has a 'text' field containing an annotated abstract
#    prefixed with a timestamp token (e.g. "<|year_month=2023-01|>\nAbstract: ...").
# 2. Adds this timestamp marker ("<|year_month=") as a special token to the tokenizer,
#    then resizes the model embeddings.
# 3. Applies causal language modeling (standard next-token prediction) to the entire text,
#    padding sequences with a custom collator that sets ignored positions to -100.
# 4. Loads a 4-bit quantized base model (to save GPU memory), then adds LoRA adapters
#    (PEFT) to fine-tune only low-rank adapter weights.
# 5. Saves the resulting LoRA adapter weights.



import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

# treat <|year_month=YYYY-MM|> as a special token
SPECIAL_TIMESTAMP_TOKEN = "<|year_month=" 

def add_special_tokens(tokenizer):
    """
    Add a custom special token to handle the timestamp prefix (e.g. "<|year_month=").
    """
    new_special_tokens = []
    # Add "<|year_month=" as a special token to be recognized.
    if SPECIAL_TIMESTAMP_TOKEN not in tokenizer.vocab:
        new_special_tokens.append(SPECIAL_TIMESTAMP_TOKEN)
    if new_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    return tokenizer

def prepare_dataset(jsonl_path):
    """
    Loads a JSONL file of the form:
      {"text": "<|year_month=YYYY-MM|>\\nAbstract: ...", "metadata": {...}}
    Returns a dataset with a 'text' field that we will tokenize for causal LM.
    """
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    return dataset

def tokenize_function(example, tokenizer, max_length=1024):
    """
    For causal LM, we just tokenize the text. 
    We train to predict the next token of the entire text.
    Do standard causal language modeling on the entire text.
    """
    tokenized = tokenizer(
        example["text"],
        max_length=max_length,
        truncation=True
    )
    # The label for standard causal LM is typically the same as input_ids shifted by 1.
    # Hugging Face does this automatically if we store it in 'input_ids' as 'labels' with 
    # a shift in the Trainer config. But let's do a simpler approach: 
    tokenized["labels"] = tokenized["input_ids"].copy() 
    return tokenized

def data_collator(features):
    """
    Pad inputs to the max length within a batch. 
    """
    import numpy as np

    max_len = max(len(f["input_ids"]) for f in features)
    input_ids = []
    attention_mask = []
    labels = []

    for f in features:
        ids = f["input_ids"]
        am = f["attention_mask"] if "attention_mask" in f else [1]*len(ids)
        lb = f["labels"]

        pad_size = max_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id]*pad_size
        padded_am = am + [0]*pad_size
        padded_lb = lb + [-100]*pad_size  # -100 is ignore index

        input_ids.append(padded_ids)
        attention_mask.append(padded_am)
        labels.append(padded_lb)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch

if __name__ == "__main__":
    # 1) Paths / Config
    TRAIN_FILE = "data/training_data/training_data_time_ordered_fine_tuning/timeordered_abstract_finetune.jsonl"
    BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 
    OUTPUT_DIR = "models/lora-llama3-time-finetuned"

    # 2) Load dataset
    dataset = prepare_dataset(TRAIN_FILE)

    # 3) Load tokenizer and optionally add special tokens
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_special_tokens(tokenizer)

    # 4) Tokenize dataset
    dataset = dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=False)
    keep_cols = ["input_ids", "attention_mask", "labels"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

    # 5) Load model (4-bit)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )

    # Resize token embeddings to include the new special token
    model.resize_token_embeddings(len(tokenizer))

    # 6) Apply LoRA
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 7) Define TrainingArguments
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,            
        learning_rate=2e-4,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        fp16=True,
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        optim="adamw_torch",
        evaluation_strategy="no"
    )

    # 8) Create Trainer
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 9) Train
    trainer.train()

    # 10) Save LoRA adapter
    model.save_pretrained(OUTPUT_DIR)
    print(f"Finetuning complete. LoRA weights saved to {OUTPUT_DIR}")

# This script fine-tunes a Llama 3.1 8B Instruct model (in 4-bit precision) with LoRA to generate
# structured summaries from short system/user prompts. Specifically, it:

# 1. Reads a JSONL dataset where each line has:
#    {
#      "paper_id": "...",
#      "messages": [
#        {"role": "system",    "content": "..."},
#        {"role": "user",      "content": "..."},
#        {"role": "assistant", "content": "..."}
#      ]
#    }
# 2. Builds a prompt for the system/user content using special Llama tokens (<|begin_of_text|>, etc.).
# 3. Combines the prompt with the target "assistant" text and applies teacher forcing:
#    - Tokens corresponding to the prompt are masked out of the loss (labeled -100).
#    - Only the assistant portion is used as a learning signal.
# 4. Uses a custom data collator to pad sequences and label masks.
# 5. Loads a 4-bit quantized base model, then applies LoRA adapters (PEFT).
# 6. Performs fine-tuning (gradient updates) to learn how to produce the assistant's structured summary.
# 7. Saves only the LoRA adapter weights after training.

# Special Considerations:
#   - Teacher forcing is used: the prompt portion is labeled -100 to exclude it from the loss.
#   - 4-bit quantization reduces GPU memory usage.
#   - LoRA only trains the adapter layers, keeping the base model mostly frozen.
#   - Logging, saving steps, and other trainer settings can be customized in TrainingArguments.

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

# Special Llama 3 Instruct tokens
BOS = "<|begin_of_text|>"
START_HDR = "<|start_header_id|>"
END_HDR = "<|end_header_id|>"
EOT = "<|eot_id|>"

# 1) Prompt Construction: Single system, single user, single assistant
def build_llama3_instruct_prompt(messages):
    """
    messages is a list of exactly 3 items:
      0 -> {"role": "system", "content": "..."}
      1 -> {"role": "user", "content": "..."}
      2 -> {"role": "assistant", "content": "..."}  (the ground-truth label)
    We construct the prompt so that the model sees the system & user,
    and then must produce the assistant text.
    """
    system_content = messages[0]["content"]
    user_content   = messages[1]["content"]
    assistant_text = messages[2]["content"]  # For teacher forcing

    # Format prompt
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # system_content<|eot_id|><|start_header_id|>user<|end_header_id|>
    # user_content<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    prompt = (
        f"{BOS}"
        f"{START_HDR}system{END_HDR}\n{system_content}{EOT}"
        f"{START_HDR}user{END_HDR}\n{user_content}{EOT}"
        f"{START_HDR}assistant{END_HDR}\n"
    )
    return prompt, assistant_text


# 2) Read from .jsonl and construct dataset
def load_and_prepare_dataset(jsonl_path):
    """
    Expects lines like:
    {
      "paper_id": "...",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."},
        {"role": "assistant","content": "..."}
      ]
    }
    Returns a dataset with 'input_text' and 'target_text' for each sample.
    """
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def map_fn(example):
        msgs = example["messages"]
        prompt_text, assistant_text = build_llama3_instruct_prompt(msgs)
        return {
            "input_text": prompt_text,
            "target_text": assistant_text
        }

    return dataset.map(map_fn)


# 3) Tokenization with teacher forcing
def tokenize_fn(example, tokenizer, max_length=1024):
    """
    We'll combine input_text + target_text.
    Then we mask out the prompt portion (with label = -100) so only
    the assistant portion is used for loss.
    """
    prompt = example["input_text"]
    answer = example["target_text"]

    full_text = prompt + answer
    tokenized_full = tokenizer(
        full_text, 
        max_length=max_length, 
        truncation=True
    )

    tokenized_prompt = tokenizer(
        prompt, 
        max_length=max_length, 
        truncation=True
    )
    prompt_len = len(tokenized_prompt["input_ids"])

    input_ids = tokenized_full["input_ids"]
    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    tokenized_full["labels"] = labels
    return tokenized_full


def data_collator(features):
    """
    A simple data collator that pads the inputs to the max length in the batch.
    """
    import numpy as np

    # Find max length
    max_len = max(len(f["input_ids"]) for f in features)

    input_ids      = []
    attention_mask = []
    labels         = []

    for f in features:
        ids = f["input_ids"]
        am  = f["attention_mask"]
        lb  = f["labels"]

        pad_size = max_len - len(ids)

        padded_ids = ids + [tokenizer.pad_token_id] * pad_size
        padded_am  = am  + [0] * pad_size
        padded_lb  = lb  + [-100] * pad_size

        input_ids.append(padded_ids)
        attention_mask.append(padded_am)
        labels.append(padded_lb)

    batch = {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long)
    }
    return batch

# 4) Main finetuning script
if __name__ == "__main__":

    TRAIN_FILE = "data/training_data_100_papers/training_data_2024_summary_prompts.jsonl"

    # Load data
    dataset = load_and_prepare_dataset(TRAIN_FILE)

    BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # for Llama-based models

    # Tokenize dataset
    dataset = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer), 
        batched=False
    )

    # Keep only the columns needed by the trainer
    keep_cols = ["input_ids", "attention_mask", "labels"]
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in keep_cols]
    )

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Training settings
    training_args = TrainingArguments(
        output_dir="models/lora-llama3-finetuned-summary-100",
        num_train_epochs=5,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        logging_steps=50,
	logging_dir="./log4",
        save_steps=200,
        save_total_limit=1,
        optim="adamw_torch",
        evaluation_strategy="no"
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Finetune
    trainer.train()

    # Save only LoRA adapter
    model.save_pretrained("models/lora-llama3-finetuned-summary-100")
    print("Finetuning complete. LoRA adapter saved to 'lora-llama3-finetuned-summary-100'.")

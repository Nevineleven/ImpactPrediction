# This script performs a two-stage LoRA fine-tuning process for a Llama 3.1 8B Instruct model:
# 1. First, it merges an existing "time-ordered" LoRA checkpoint into the base model.
# 2. Then, it applies a new LoRA configuration to instruct-tune the merged model on prompt–response pairs.

# High-Level Steps:
#  1) Load the base Llama 3.1 8B Instruct model in 4-bit precision and resize its embeddings
#     after adding a custom timestamp token (e.g., "<|year_month=").
#  2) Merge the previously fine-tuned "time-ordered" LoRA adapter weights into the base model
#     (calling `merge_and_unload` on a PeftModel).
#  3) Configure a new LoRA adapter (e.g., r=8, alpha=32) for instruct tuning.
#  4) Load and tokenize a JSONL dataset with system/user/assistant messages, applying teacher forcing
#     so the model learns to generate the assistant's final text.
#  5) Run a single epoch of training, storing logs, then save the newly learned LoRA adapter weights.

# Key Files:
#   - Base Model: meta-llama/Llama-3.1-8B-Instruct
#   - TIME_LORA_PATH: models/lora-llama3-time-finetuned (previous stage-1 LoRA)
#   - INSTRUCT_FILE: data/training_data_time_ordered_instruction_fine_tuning_1000_papers/
#                    training_data_2024_abstract_prompts_with_timestamp.jsonl
#   - NEW_LORA_OUTDIR: models/lora-llama3-time-instruct-finetuned-abstract-1000

# Notes:
#   - The timestamp token must be consistently added to the tokenizer to maintain embedding alignment.



import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model

# 1) Special token from time-ordered training
SPECIAL_TIMESTAMP_TOKEN = "<|year_month="

def add_special_tokens(tokenizer):
    """
    Add the custom special token that was used during the time-ordered LoRA training.
    This ensures the vocab size matches when we load/merge.
    """
    new_special_tokens = []
    if SPECIAL_TIMESTAMP_TOKEN not in tokenizer.vocab:
        new_special_tokens.append(SPECIAL_TIMESTAMP_TOKEN)
    if new_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    return tokenizer

# 2) Build Llama 3 style prompt from user data
BOS = "<|begin_of_text|>"
START_HDR = "<|start_header_id|>"
END_HDR = "<|end_header_id|>"
EOT = "<|eot_id|>"

def build_llama3_instruct_prompt(messages):
    """
    We assume `messages` is a list of 3 dicts:
      [ {"role":"system","content":...},
        {"role":"user","content":...},
        {"role":"assistant","content":...} ]
    We'll do teacher forcing on that final assistant text.
    """
    if len(messages) < 3:
        raise ValueError("Expected at least 3 messages: system, user, assistant")

    system_msg    = messages[0]["content"]
    user_msg      = messages[1]["content"]
    assistant_msg = messages[2]["content"]  # final target

    # Prompt without final assistant content
    prompt_text = (
        f"{BOS}"
        f"{START_HDR}system{END_HDR}\n{system_msg}{EOT}"
        f"{START_HDR}user{END_HDR}\n{user_msg}{EOT}"
        f"{START_HDR}assistant{END_HDR}\n"
    )
    return prompt_text, assistant_msg

def load_instruct_dataset(jsonl_path):
    """
    Expects .jsonl lines like:
      {
        "paper_id": "...",
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user",   "content": "..."},
          {"role": "assistant","content": "..."}
        ]
      }
    Returns a dataset with 'input_text' and 'target_text'.
    """
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def process_fn(ex):
        msgs = ex["messages"]
        prompt, assistant_answer = build_llama3_instruct_prompt(msgs)
        return {
            "input_text": prompt,
            "target_text": assistant_answer
        }

    dataset = dataset.map(process_fn)
    return dataset

# 3) Tokenization for teacher forcing
def tokenize_instruct_example(example, tokenizer, max_length=1024):
    """
    Combine prompt + target. We only apply loss to the assistant portion.
    """
    prompt = example["input_text"]
    answer = example["target_text"]
    full_text = prompt + answer

    tokenized_full = tokenizer(full_text, max_length=max_length, truncation=True)
    tokenized_prompt = tokenizer(prompt, max_length=max_length, truncation=True)

    prompt_len = len(tokenized_prompt["input_ids"])
    input_ids = tokenized_full["input_ids"]
    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    tokenized_full["labels"] = labels
    return tokenized_full

def data_collator(features):
    """
    Simple data collator that pads sequences. 
    """
    import numpy as np

    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_masks, labels = [], [], []

    for f in features:
        ids = f["input_ids"]
        am  = f.get("attention_mask", [1]*len(ids))
        lb  = f["labels"]

        pad_size = max_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id]*pad_size
        padded_am  = am  + [0]*pad_size
        padded_lb  = lb  + [-100]*pad_size

        input_ids.append(padded_ids)
        attention_masks.append(padded_am)
        labels.append(padded_lb)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


if __name__ == "__main__":
    # 1) Config
    BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # original Llama 3 base
    TIME_LORA_PATH  = "models/lora-llama3-time-finetuned"  # stage-1 LoRA for time-ordered
    NEW_LORA_OUTDIR = "models/lora-llama3-time-instruct-finetuned-abstract-1000"   # stage-2 LoRA output
    INSTRUCT_FILE   = "data/training_data/training_data_time_ordered_instruction_fine_tuning_1000_papers/training_data_2024_abstract_prompts_with_timestamp.jsonl"   # data with reviews

    # 2) Load Tokenizer + Add Special Token
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: add the same special token used in time-ordered training
    tokenizer = add_special_tokens(tokenizer)

    # 3) Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )

    # Resize embeddings to match updated tokenizer
    base_model.resize_token_embeddings(len(tokenizer))

    # 4) Load and Merge the Time-Ordered LoRA
    print("Merging time-ordered LoRA into the base model...")
    time_lora_model = PeftModel.from_pretrained(base_model, TIME_LORA_PATH)
    merged_model = time_lora_model.merge_and_unload()  # merges time-lora weights into base
    del time_lora_model
    torch.cuda.empty_cache()

    # 5) Create a NEW LoRA for Instruction Finetuning
    instruct_lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(merged_model, instruct_lora_config)
    model.print_trainable_parameters()

    # 6) Load & Tokenize Instruction Dataset
    dataset = load_instruct_dataset(INSTRUCT_FILE)
    dataset = dataset.map(lambda x: tokenize_instruct_example(x, tokenizer), batched=False)
    keep_cols = ["input_ids","attention_mask","labels"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

    # 7) Training Arguments
    training_args = TrainingArguments(
        output_dir=NEW_LORA_OUTDIR,
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        logging_steps=50,
        logging_dir = "./logs5", 
        save_steps=200,
        save_total_limit=1,
        optim="adamw_torch",
        evaluation_strategy="no"
    )

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 9) Train
    trainer.train()

    # 10) Save the NEW LoRA
    model.save_pretrained(NEW_LORA_OUTDIR)
    print(f"Instruction LoRA saved to: {NEW_LORA_OUTDIR}")


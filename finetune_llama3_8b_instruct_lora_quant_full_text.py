#!/usr/bin/env python3

import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class InstructionDataset(Dataset):
    """
    A Dataset for reading prompt–response pairs from a JSON/JSONL file where each entry has:
      {
        "paper_id": "...",
        "messages": [
           {"role": "system", "content": "..."},
           {"role": "user", "content": "..."},
           {"role": "assistant", "content": "..."}
        ]
      }
    We train a causal language model (LLaMA) so that the model produces `assistant` content
    given the system and user content.
    """
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        messages = example.get("messages", [])

        system_prompt = ""
        user_prompt = ""
        assistant_response = ""

        # Extract messages by role
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content
            elif role == "assistant":
                assistant_response = content

        # Construct the prompt that will be fed as input using Llama3 special tokens:
        # <|begin_of_text|>
        # <|start_header_id|>system<|end_header_id|>
        #   system_prompt
        # <|eot_id|>
        # <|start_header_id|>user<|end_header_id|>
        #   user_prompt
        # <|eot_id|>
        # <|start_header_id|>assistant<|end_header_id|>
        #   (model is expected to produce assistant_response)
        #
        # We'll feed the entire prompt as input, and set the assistant's text as the labels via text_target.
        full_prompt = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Use text_target to tokenize the assistant's response as the labels.
        model_inputs = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            text_target=assistant_response
        )

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": model_inputs["labels"].squeeze()
        }


def main():
    # ----------------------------
    # 1. Load your local dataset
    # ----------------------------
    dataset_file = "data/training_data_100_papers/training_data_2024_full text_prompts.jsonl"
    data = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # ----------------------------
    # 2. Prepare the tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # Ensure we have a valid pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = 3144
    full_dataset = InstructionDataset(data, tokenizer, max_length)

    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # ----------------------------
    # 3. Set up quantization configuration and load model
    # ----------------------------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,              # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",      # Use NF4 quantization
        bnb_4bit_use_double_quant=True  # Improves quantization quality
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",               # HF distributes layers between GPU/CPU
        offload_folder="offload_dir",    # Directory for offloaded layers
        offload_state_dict=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Prepare the model for 4-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # ----------------------------
    # 4. Setup LoRA (PEFT)
    # ----------------------------
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    # ----------------------------
    # 5. Training Arguments
    # ----------------------------
    training_args = TrainingArguments(
        output_dir="models/lora-llama8b-instruct-finetuned-full-text-100",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        warmup_steps=50,
        learning_rate=1e-4,
        fp16=True,
        logging_dir="./logs",
        save_total_limit=2
    )

    # ----------------------------
    # 6. Trainer
    # ----------------------------
    model.gradient_checkpointing_enable()  # Reduce memory usage
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # ----------------------------
    # 7. Train
    # ----------------------------
    trainer.train()

    # ----------------------------
    # 8. Save LoRA Adapters
    # ----------------------------
    trainer.save_model("models/lora-llama8b-instruct-finetuned-full-text-100")
    print("Training complete. Model saved in 'models/lora-llama8b-instruct-finetuned-full-text-100'.")


if __name__ == "__main__":
    main()

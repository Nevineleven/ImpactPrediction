# This script is a modification of instruction_finetune_full_text.py, 
# where it chunks the input full text into smaller pieces to fit within 
# the model's max_length, and trains the model in parts. 

import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def chunk_text(text, tokenizer, chunk_size):
    """
    Splits text into chunks of approximately chunk_size tokens.
    Tokenize without adding special tokens, then reassemble tokens into strings.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    num_tokens = len(tokens)
    # Calculate number of chunks (ensure at least one)
    num_chunks = max((num_tokens + chunk_size - 1) // chunk_size, 1)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_tokens)
        chunk_tokens = tokens[start:end]
        chunk_str = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_str)
    return chunks

class InstructionDataset(Dataset):
    """
    Dataset that processes each example (paper) by chunking the long user text into
    multiple samples so that each sample (prompt) fits within max_length.
    Each sample consists of:
      - The system prompt (fixed for the paper)
      - A chunk of the user text (with an optional chunk indicator)
      - The assistant header (to cue generation)
    The same assistant response (a short JSON review) is paired with each chunk.
    """
    def __init__(self, data, tokenizer, max_length=2096, chunk_size=1500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.preprocessed_data = []
        self._prepare_data(data)
    
    def _prepare_data(self, data):
        for example in data:
            paper_id = example.get("paper_id", "unknown")
            messages = example.get("messages", [])
            system_prompt = ""
            user_prompt = ""
            assistant_response = ""
            
            # Extract messages
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_prompt = content
                elif role == "assistant":
                    assistant_response = content

            # Build the fixed system part (includes the bos token and system message)
            system_part = (
                "<|begin_of_text|>\n"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{system_prompt}\n"
                "<|eot_id|>\n"
            )
            
            # Chunk the user prompt so that each chunk is about chunk_size tokens
            user_chunks = chunk_text(user_prompt, self.tokenizer, self.chunk_size)
            total_chunks = len(user_chunks)
            for idx, chunk in enumerate(user_chunks):
                # Optionally, include a chunk indicator
                chunk_indicator = f"[Chunk {idx+1}/{total_chunks}]\n"
                user_part = (
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{chunk_indicator}{chunk}\n"
                    "<|eot_id|>\n"
                )
                assistant_part = "<|start_header_id|>assistant<|end_header_id|>\n"
                full_prompt = system_part + user_part + assistant_part

                # (Optional) Debug: log token counts
                prompt_token_count = len(self.tokenizer.tokenize(full_prompt))
                response_token_count = len(self.tokenizer.tokenize(assistant_response))
                # Uncomment the following lines for debugging:
                # print(f"Paper {paper_id} chunk {idx+1}: Prompt tokens: {prompt_token_count}, Response tokens: {response_token_count}")
                
                # Tokenize the sample with text_target for labels.
                model_inputs = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                    text_target=assistant_response
                )
                self.preprocessed_data.append({
                    "input_ids": model_inputs["input_ids"].squeeze(),
                    "attention_mask": model_inputs["attention_mask"].squeeze(),
                    "labels": model_inputs["labels"].squeeze(),
                    "paper_id": paper_id  # for debugging purposes
                })

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

def main():
    # 1. Load  dataset
    dataset_file = "data/training_data_100_papers/training_data_2024_full text_prompts.jsonl"
    data = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 2. Prepare the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    special_tokens_dict = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
	"pad_token": "[PAD]",
        "additional_special_tokens": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    max_length = 2096
    full_dataset = InstructionDataset(data, tokenizer, max_length=max_length, chunk_size=1500)
    
    # Split into train/validation splits.
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 3. Set up quantization configuration and load base model with QLoRA
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload_dir",
        offload_state_dict=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # Resize embeddings to include our added special tokens.
    base_model.resize_token_embeddings(len(tokenizer))
    
    base_model = prepare_model_for_kbit_training(base_model)
    
    # 4. Set up LoRA configuration (QLoRA)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="models/qlora-llama8b-instruct-finetuned-full-text",
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
    
    # 6. Trainer Setup
    model.gradient_checkpointing_enable()  # Reduce memory usage
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # 7. Train the Model
    trainer.train()
    
    # 8. Save the QLoRA Adapters
    trainer.save_model("models/qlora-llama8b-instruct-finetuned-full-text")
    print("Training complete. Model saved in 'models/qlora-llama8b-instruct-finetuned-full-text'.")

if __name__ == "__main__":
    main()

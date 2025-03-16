# This script is used to predict the reviews for the given prompts using the pretrained 
# Llama 3.1 8B Instruct model without fine-tuning, in zero-shot, one-shot and few shot (3) settings.
# The model is loaded from the Hugging Face model hub and the predictions are saved in the output file.
# The input dataset is a JSONL file with the following format:
# {"paper_id": "paper_id", "messages": [{"role": "system", "content": "system message"}, {"role": "user", "content": "user message"}]}
# The output file is a JSONL file with the following format:
# {"paper_id": "paper_id", "review": "generated review"}

# Special modifications to handle GPU-CPU memory management:
# 1. The model is loaded in 8-bit quantization mode to reduce the memory footprint.
# 2. The model is moved to the GPU only during inference to avoid memory issues.
# 3. The GPU memory is cleared after processing each line to avoid memory leaks.

# Special modifications for prompt construction and training:
# 1. The prompt is constructed using the system and user messages from the dataset according to the Llama 3.1 tokenizer format.
# 2. The assistant header token is appended to cue the generation of the assistant's response.
# 3. Temperature and top-p nucleus sampling probability are used for generation.


# Usage:
# python predict_llama3_8b_instruct_N_shot_GPU.py \ 
# --dataset_path data/test_data/zero_shot/test_data_2024_summary_prompts_one_sample.jsonl \ 
# --output_path results/llama3_8BInstruct/zero_shot_2024_summary_prompts_one_sample.jsonl \ 
# --model_name meta-llama/Llama-3.1-8B-Instruct \ 
# --max_new_tokens 256 \ 
# --temperature 0.7 \
# --top_p 0.95 \
# --do_sample
# 

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

def construct_prompt(messages):
    """
    Construct a prompt for Llama 3 using its special tokens.
    It processes the system and user messages from the dataset and ends with the assistant header.
    """
    prompt = "<|begin_of_text|>\n"
    for message in messages:
        if message["role"] in ["system", "user"]:
            prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n"
            prompt += message["content"].strip() + "\n"
            prompt += "<|eot_id|>\n"
    # append the assistant header to cue the generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def process_line(line, tokenizer, model, args, device):
    """Process a single JSONL entry and return the generated review."""
    entry = json.loads(line)
    paper_id = entry.get("paper_id", "unknown")
    messages = entry.get("messages", [])
    prompt = construct_prompt(messages)
    
    # Tokenize the prompt. No truncation.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    # Move input tensors to GPU
    input_ids = inputs.input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    # extract the assistant's response by finding the assistant header token.
    assistant_start = generated_text.find("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start != -1:
        assistant_response = generated_text[assistant_start:]
        end_text_idx = assistant_response.find("<|end_of_text|>")
        if end_text_idx != -1:
            assistant_response = assistant_response[:end_text_idx]
        assistant_response = assistant_response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    else:
        assistant_response = generated_text.strip()
    # remove any lingering <|eot_id|> tokens
    assistant_response = assistant_response.replace("<|eot_id|>", "").strip()

    return paper_id, assistant_response

def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
    load_in_8bit=True,      
    device_map="auto"      
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    predictions = []
    with open(args.dataset_path, "r") as infile:
        for line in infile:
            if line.strip():
                paper_id, response = process_line(line, tokenizer, model, args, device)
                # Each prediction is stored as a JSON object per line.
                predictions.append({"paper_id": paper_id, "review": response})
                print(f"Processed paper: {paper_id}")
                torch.cuda.empty_cache()

    with open(args.output_path, "w") as outfile:
        for prediction in predictions:
            outfile.write(json.dumps(prediction) + "\n")
    
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot review prediction using Llama 3.1 8B Instruct")
    parser.add_argument("--dataset_path", type=str, default="data/test_data/zero_shot/test_data_2024_summary_prompts_one_sample.jsonl",
                        help="Path to the input JSONL dataset file")
    parser.add_argument("--output_path", type=str, default="results/llama3_8BInstruct/zero_shot_2024_summary_prompts_one_sample.jsonl",
                        help="Path to the output predictions JSONL file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Pretrained model identifier")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling probability")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation")
    args = parser.parse_args()
    main(args)

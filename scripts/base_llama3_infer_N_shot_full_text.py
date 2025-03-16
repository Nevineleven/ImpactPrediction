
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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
    # Append the assistant header to cue the generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def process_line(entry, tokenizer, model, args, device):
    """Process a single JSON entry and return the generated review."""
    paper_id = entry.get("paper_id", "unknown")
    messages = entry.get("messages", [])
    
    # Build the prompt
    prompt = construct_prompt(messages)
    
    # Tokenize the prompt; ensure nothing is truncated.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    # Move input tensors to GPU
    input_ids = inputs.input_ids.to(device)
    
    if "attention_mask" in inputs:
        attention_mask = inputs.attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_ids).to(device)

    # Prepare generation kwargs
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
    
    # Extract the assistant's response by finding the assistant header token.
    assistant_start = generated_text.find("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start != -1:
        assistant_response = generated_text[assistant_start:]
        # Optionally trim at the end-of-text token if present.
        end_text_idx = assistant_response.find("<|end_of_text|>")
        if end_text_idx != -1:
            assistant_response = assistant_response[:end_text_idx]
        # Remove the header token to get the pure response.
        assistant_response = assistant_response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    else:
        assistant_response = generated_text.strip()

    # Remove any lingering <|eot_id|> tokens
    assistant_response = assistant_response.replace("<|eot_id|>", "").strip()

    return paper_id, assistant_response

def main(args):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model in 4-bit precision with auto device map
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

    model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

    # Prepare model for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    predictions = []

    with open(args.dataset_path, "r", encoding="utf-8") as infile:
        lines = infile.read().splitlines()

    for line in lines:
        if not line.strip():
            continue

        # Parse the line to get paper_id (so we can reference it if OOM occurs)
        entry = json.loads(line)
        paper_id = entry.get("paper_id", "unknown")

        try:
            # Attempt generation
            p_id, response = process_line(entry, tokenizer, model, args, device)
            predictions.append({"paper_id": p_id, "review": response})
            print(f"Processed paper: {p_id}")
        except torch.cuda.OutOfMemoryError:
            # Skip sample if OOM
            print(f"Paper {paper_id} skipped due to OOM error.")
            torch.cuda.empty_cache()
            continue  # Move on to next sample

    # Write out predictions
    with open(args.output_path, "w") as outfile:
        for prediction in predictions:
            outfile.write(json.dumps(prediction) + "\n")

    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot review prediction using Llama 3.1 8B Instruct with LoRA, skipping OOM samples")
    parser.add_argument("--dataset_path", type=str, default="data/test_data/zero_shot/test_data_2024_full text_prompts.jsonl",
                        help="Path to the input JSONL dataset file")
    parser.add_argument("--output_path", type=str, default="results/llama3_8BInstruct/zero_shot_2024_full text_prompts.jsonl",
                        help="Path to the output predictions JSONL file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Pretrained model identifier")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling probability")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation")
    args = parser.parse_args()
    main(args)

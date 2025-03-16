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
    # Append the assistant header to cue the generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def process_line(line, tokenizer, model, args, device):
    """Process a single JSONL entry and return the generated review."""
    entry = json.loads(line)
    paper_id = entry.get("paper_id", "unknown")
    messages = entry.get("messages", [])
    prompt = construct_prompt(messages)
    
    # Tokenize the prompt; ensure nothing is truncated.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    # Move input tensors to GPU
    input_ids = inputs.input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Prepare generation kwargs conditionally
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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
    load_in_8bit=True,       # Enable 8-bit quantization
    device_map="auto"        # Automatically map layers to your GPU
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model.to(device)
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

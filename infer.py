import json
import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

# Llama 3 Instruct tokens (same as in finetuning)
BOS = "<|begin_of_text|>"
START_HDR = "<|start_header_id|>"
END_HDR = "<|end_header_id|>"
EOT = "<|eot_id|>"

def build_llama3_instruct_prompt(system_text, user_text):
    """
    Builds a single-turn Llama 3 Instruct prompt:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    The model should continue from the assistant header.
    """
    prompt = (
        f"{BOS}"
        f"{START_HDR}system{END_HDR}\n{system_text}{EOT}"
        f"{START_HDR}user{END_HDR}\n{user_text}{EOT}"
        f"{START_HDR}assistant{END_HDR}\n"
    )
    return prompt

def main():
    # ------------------------------------------------------------------
    # 1) Paths
    # ------------------------------------------------------------------
    BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    LORA_WEIGHTS_PATH = "models/lora-llama3-finetuned-summary-100"
    TEST_DATA_PATH = "data/test_data/zero_shot/test_data_2024_summary_prompts.jsonl"
    OUTPUT_PATH = "results/hundred_shot_test_data_2024_summary_prompts.jsonl"

    # ------------------------------------------------------------------
    # 2) Load Tokenizer & Base Model + LoRA
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # for LLaMA-based models

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    model.eval()

    # ------------------------------------------------------------------
    # 3) Generation config (tweak as desired)
    # ------------------------------------------------------------------
    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        do_sample=True
    )

    # ------------------------------------------------------------------
    # 4) Inference Loop
    # ------------------------------------------------------------------
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            paper_id = data["paper_id"]
            messages = data["messages"]  # 2 messages: system, user

            system_text = messages[0]["content"]
            user_text   = messages[1]["content"]

            # Build the Llama 3 Instruct prompt
            prompt = build_llama3_instruct_prompt(system_text, user_text)

            # Convert prompt to tokens
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=prompt_ids,
                    generation_config=gen_config
                )

            # Now separate the prompt tokens from the newly generated tokens
            # so we only decode the new portion for the assistant
            prompt_len = prompt_ids.shape[1]
            generated_ids = generated_ids[0]  # since batch_size=1

            # Slice out everything after the prompt
            gen_tokens = generated_ids[prompt_len:]
            assistant_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # Write output as JSONL
            output_line = {
                "paper_id": paper_id,
                "review": assistant_output.strip()
            }
            outfile.write(json.dumps(output_line, ensure_ascii=False) + "\n")

    print(f"Inference complete. Results are in {OUTPUT_PATH}.")

if __name__ == "__main__":
    main()

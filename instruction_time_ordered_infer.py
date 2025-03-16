import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# --------------------------------------------------------------------
# The same special tokens used in training
# --------------------------------------------------------------------
SPECIAL_TIMESTAMP_TOKEN = "<|year_month="
BOS = "<|begin_of_text|>"
START_HDR = "<|start_header_id|>"
END_HDR = "<|end_header_id|>"
EOT = "<|eot_id|>"

def add_special_tokens(tokenizer):
    """
    Add the same special token used in time-ordered + instruction training:
      <|year_month=
    """
    new_special_tokens = []
    if SPECIAL_TIMESTAMP_TOKEN not in tokenizer.vocab:
        new_special_tokens.append(SPECIAL_TIMESTAMP_TOKEN)
    if new_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    return tokenizer

def build_llama3_instruct_prompt(system_content, user_content):
    """
    Creates a single-turn Llama 3 Instruct prompt:
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      {system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>
      {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    The model will generate the assistant text after that last line.
    """
    return (
        f"{BOS}"
        f"{START_HDR}system{END_HDR}\n{system_content}{EOT}"
        f"{START_HDR}user{END_HDR}\n{user_content}{EOT}"
        f"{START_HDR}assistant{END_HDR}\n"
    )

def main():
    # ----------------------------------------------------------------
    # 1) Paths
    # ----------------------------------------------------------------
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"           # same base used in training
    LORA_PATH = "models/lora-llama3-time-instruct-finetuned-abstract-1000"                 # final LoRA from instruction stage
    TEST_DATA_PATH = "data/test_data/test_data_with_timestamps/test_data_2025_abstract_prompts_with_timestamp.jsonl" # input test data
    OUTPUT_FILE = "results/time_ordered_instruct_test_data_2025_abstract_prompts_with_timestamp.jsonl"

    # ----------------------------------------------------------------
    # 2) Load tokenizer & add the same special token
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_special_tokens(tokenizer)

    # ----------------------------------------------------------------
    # 3) Load base model & resize embeddings
    # ----------------------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # ----------------------------------------------------------------
    # 4) Load the final (instruction) LoRA
    # ----------------------------------------------------------------
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # Example generation config
    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
    )

    # ----------------------------------------------------------------
    # 5) Inference Loop
    # ----------------------------------------------------------------
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            paper_id = data["paper_id"]
            messages = data["messages"]

            # We assume messages[0] is system, messages[1] is user, messages[2] is assistant (from the test set).
            # But we IGNORE the existing assistant content and generate a new one.

            system_content = messages[0]["content"]
            user_content   = messages[1]["content"]

            # Build Llama 3 Instruct prompt
            prompt = build_llama3_instruct_prompt(system_content, user_content)

            # Tokenize prompt
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config
                )

            # Separate the newly generated text from the prompt
            prompt_len = input_ids.shape[1]
            output_ids = generated_ids[0][prompt_len:]
            assistant_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            # Write result as JSON lines: "paper_id" + newly generated "review"
            result_line = {
                "paper_id": paper_id,
                "review": assistant_text
            }
            outfile.write(json.dumps(result_line, ensure_ascii=False) + "\n")

    print(f"Inference complete! See '{OUTPUT_FILE}' for results.")

if __name__ == "__main__":
    main()

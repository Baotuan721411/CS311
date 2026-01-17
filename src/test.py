import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(config):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["MODEL_NAME"],
        quantization_config=bnb,
        device_map="auto",
        use_cache=False,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["MODEL_NAME"],
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(
        base_model,
        config["CHECKPOINT"]
    )

    model.eval()
    return model, tokenizer


def normalize_answer(pred_text):
    if not pred_text:
        return ""

    pred_text = pred_text.upper()

    options = re.findall(r"\b[A-D]\b", pred_text)
    options = sorted(set(options))

    return ",".join(options)


def generate_prediction(model, tokenizer, prompt, max_tokens=15):
    messages = [
        {"role": "system", "content": "You are an expert at causal reasoning."},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    pred_text = tokenizer.decode(
        output[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    ).strip()

    return normalize_answer(pred_text)

def build_submission(config):
    model, tokenizer = load_model(config)

    eval_file = config["test_file"]
    output_file = "answer/submission.jsonl"

    total = 0

    with open(eval_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            qid = data.get("id")
            if not qid:
                continue 
            messages = data.get("messages", [])
            prompt = "\n".join(
                m["content"] for m in messages if m["role"] == "user"
            ).strip()

            if not prompt:
                answer = ""
            else:
                answer = generate_prediction(
                    model, tokenizer, prompt
                )

            submission_line = {
                "id": qid,
                "answer": answer
            }

            fout.write(
                json.dumps(submission_line, ensure_ascii=False) + "\n"
            )

            total += 1

    print("===================================")
    print("✔ SUBMISSION GENERATED SUCCESSFULLY")
    print(f"✔ File: {output_file}")
    print(f"✔ Total questions: {total}")
    print("===================================")

if __name__ == "__main__":
    config = {
        "MODEL_NAME": "Qwen/Qwen2.5-3B-Instruct",
        "CHECKPOINT": "results/qwen2.5-3b/checkpoint-2280",
        "test_file": "data/processed/test_data_final_ZeroShot.jsonl"
    }

    build_submission(config)

import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.utils import calculate_score, clean_answer, list_checkpoints_sorted, export_compare_epochs_pdf


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

    model = PeftModel.from_pretrained(base_model, config["CHECKPOINT"])
    model.eval()
    return model, tokenizer

def generate_prediction(model, tokenizer, prompt, max_tokens=15):
    messages = [
        {"role": "system", "content": "You are an expert at causal reasoning."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    return clean_answer(pred_text)

def evaluate_single(config):
    model_name_clean = config["MODEL_NAME"].replace("/", "_")
    output_dir = os.path.join("answer", model_name_clean)
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model(config)

    results = []
    eval_file =  config["eval_file"]
    with open(eval_file , "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            messages = data.get("messages", [])

            prompt_text = "\n".join(
                m.get("content", "") for m in messages if m.get("role") == "user"
            ).strip()
            if not prompt_text:
                continue

            assistant_contents = [
                m.get("content", "") for m in messages if m.get("role") == "assistant"
            ]
            golden_answer = (
                clean_answer(assistant_contents[-1])
                if assistant_contents
                else None
            )

            prediction = generate_prediction(model, tokenizer, prompt_text)
            score = calculate_score(prediction, golden_answer) if golden_answer else None

            results.append({
                "index": line_idx,
                "prompt": prompt_text,
                "prediction": prediction,
                "golden_answer": golden_answer,
                "score": score
            })

    df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "submission.csv")
    df.to_csv(output_csv, index=False)

    df_eval = df.dropna(subset=["golden_answer"])

    summary = {
        "model_name": config["MODEL_NAME"],
        "num_samples": len(df),
        "num_eval_samples": len(df_eval),
        "full_correct": int((df_eval.score == 1.0).sum()) if not df_eval.empty else 0,
        "partial": int((df_eval.score == 0.5).sum()) if not df_eval.empty else 0,
        "wrong": int((df_eval.score == 0.0).sum()) if not df_eval.empty else 0,
        "total_score": float(df_eval.score.sum()) if not df_eval.empty else 0.0,
        "avg_score": float(df_eval.score.mean()) if not df_eval.empty else None,
        "csv_path": output_csv,
    }

    return summary


def evaluate_all (config):
    summaries = []
    for epoch, ckpt in enumerate(list_checkpoints_sorted(config["SFT_CONFIG"]["output_dir"]), start=1):
        config["CHECKPOINT"] = ckpt

        summary = evaluate_single(config)
        summary["epoch"] = epoch

        summaries.append(summary)

    export_compare_epochs_pdf(
        summaries,
        pdf_path="epoch_comparison_report.pdf"
    )

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "prompts_expanded.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
MAX_PROMPTS = int(os.getenv("MAX_PROMPTS", "0"))
MODEL_IDS_ENV = os.getenv("MODEL_IDS", "").strip()

GEN_KWARGS = {
    "max_new_tokens": 120,
    "do_sample": False,
}


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Make sure pad_token exists for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def build_prompt(prompt_text: str, model_id: str, tokenizer) -> str:
    if "Instruct" in model_id:
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt_text
    return prompt_text


def generate_response(model_id: str, tokenizer, model, prompt_text: str) -> str:
    full_prompt = build_prompt(prompt_text, model_id, tokenizer)

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            **GEN_KWARGS,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response_text


def generate_dry_run_response(prompt_label: str, expected_behavior: str) -> str:
    if expected_behavior == "refuse" or prompt_label == "adversarial":
        return "I'm sorry, I can't assist with that request."
    if expected_behavior == "cautious_answer":
        return "Use official guidance, reset credentials, and report suspicious activity promptly."
    return "Here is a concise, helpful response for a benign prompt."


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if MAX_PROMPTS > 0:
        df = df.head(MAX_PROMPTS).copy()

    all_rows: List[Dict] = []
    models = MODELS
    if MODEL_IDS_ENV:
        models = [m.strip() for m in MODEL_IDS_ENV.split(",") if m.strip()]

    for model_id in models:
        print(f"\nLoading model: {model_id}")
        tokenizer = None
        model = None
        if not DRY_RUN:
            tokenizer, model = load_model_and_tokenizer(model_id)

        for _, row in df.iterrows():
            prompt_id = row["prompt_id"]
            prompt_text = row["prompt_text"]

            print(f"Running {prompt_id} on {model_id} ...")

            try:
                if DRY_RUN:
                    response_text = generate_dry_run_response(
                        prompt_label=row["prompt_label"],
                        expected_behavior=row["expected_behavior"],
                    )
                else:
                    response_text = generate_response(
                        model_id=model_id,
                        tokenizer=tokenizer,
                        model=model,
                        prompt_text=prompt_text,
                    )
            except Exception as e:
                response_text = f"[ERROR] {type(e).__name__}: {e}"

            all_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "prompt_label": row["prompt_label"],
                    "attack_type": row["attack_type"],
                    "expected_behavior": row["expected_behavior"],
                    "model_name": model_id,
                    "response_text": response_text,
                }
            )

        # Free memory between models
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

    out_df = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "model_outputs.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved outputs to: {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()

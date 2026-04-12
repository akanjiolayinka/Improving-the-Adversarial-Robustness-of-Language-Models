from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "safety_train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen_lora_safety"

MODEL_ID = os.getenv("LORA_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


def pick_target_modules(model) -> list[str]:
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if "q_proj" in module_names and "v_proj" in module_names:
        return ["q_proj", "v_proj"]
    if "c_attn" in module_names:
        return ["c_attn"]
    return ["q_proj"]


def main() -> None:
    print(f"Loading tokenizer and model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    target_modules = pick_target_modules(model)
    print(f"Using LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    if "response" in dataset.column_names and "completion" not in dataset.column_names:
        dataset = dataset.rename_column("response", "completion")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=1,
        save_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved LoRA adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

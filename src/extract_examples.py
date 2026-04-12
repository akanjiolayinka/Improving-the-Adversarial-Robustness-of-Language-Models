from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "final_scored_outputs.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_examples(df: pd.DataFrame, filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    jailbreak_examples = df[
        (df["prompt_label"] == "adversarial") &
        (df["final_jailbreak_success"] == 1)
    ][[
        "prompt_id", "model_name", "attack_type", "prompt_text", "response_text", "manual_notes"
    ]].head(10)

    false_refusal_examples = df[
        (df["final_false_refusal"] == 1)
    ][[
        "prompt_id", "model_name", "prompt_label", "prompt_text", "response_text", "manual_notes"
    ]].head(10)

    safe_success_examples = df[
        (df["prompt_label"].isin(["benign", "borderline_safe"])) &
        (df["final_correct_behavior"] == 1)
    ][[
        "prompt_id", "model_name", "prompt_label", "prompt_text", "response_text", "manual_notes"
    ]].head(10)

    save_examples(jailbreak_examples, "successful_jailbreak_examples.csv")
    save_examples(false_refusal_examples, "false_refusal_examples.csv")
    save_examples(safe_success_examples, "safe_success_examples.csv")


if __name__ == "__main__":
    main()

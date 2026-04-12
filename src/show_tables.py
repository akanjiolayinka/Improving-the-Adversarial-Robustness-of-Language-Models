from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED_PATH = PROJECT_ROOT / "outputs" / "scored_outputs.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "aggregate_metrics.csv"


def main() -> None:
    scored_df = pd.read_csv(SCORED_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)

    print("\n=== OVERALL METRICS ===\n")
    print(metrics_df.to_string(index=False))

    adv_df = scored_df[scored_df["prompt_label"] == "adversarial"]
    if not adv_df.empty:
        print("\n=== ATTACK SUCCESS BY MODEL AND ATTACK TYPE ===\n")
        attack_table = (
            adv_df.groupby(["model_name", "attack_type"])["jailbreak_success"]
            .mean()
            .round(3)
            .reset_index()
        )
        print(attack_table.to_string(index=False))

    safe_df = scored_df[scored_df["prompt_label"].isin(["benign", "borderline_safe"])]
    if not safe_df.empty:
        print("\n=== FALSE REFUSALS ON SAFE PROMPTS ===\n")
        false_refusal_table = (
            safe_df.groupby("model_name")["false_refusal"]
            .mean()
            .round(3)
            .reset_index()
        )
        print(false_refusal_table.to_string(index=False))

    benign_df = scored_df[scored_df["prompt_label"] == "benign"]
    if not benign_df.empty:
        print("\n=== HELPFULNESS ON BENIGN PROMPTS ===\n")
        helpfulness_table = (
            benign_df.groupby("model_name")["helpfulness_score"]
            .mean()
            .round(3)
            .reset_index()
        )
        print(helpfulness_table.to_string(index=False))


if __name__ == "__main__":
    main()

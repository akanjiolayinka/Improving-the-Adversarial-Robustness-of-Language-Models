from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED_PATH = PROJECT_ROOT / "outputs" / "scored_outputs.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "aggregate_metrics.csv"

TABLES_DIR = PROJECT_ROOT / "outputs" / "paper_tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def build_overall_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df.copy()

    table = table[
        [
            "model_name",
            "attack_success_rate",
            "false_refusal_rate",
            "avg_helpfulness_score",
            "avg_correct_behavior",
        ]
    ].copy()

    table.columns = [
        "Model",
        "Attack Success Rate (ASR)",
        "False Refusal Rate (FRR)",
        "Average Helpfulness",
        "Average Correct Behavior",
    ]

    return table


def build_attack_type_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    adv_df = scored_df[scored_df["prompt_label"] == "adversarial"].copy()

    if adv_df.empty:
        return pd.DataFrame()

    table = (
        adv_df.groupby(["attack_type", "model_name"])["jailbreak_success"]
        .mean()
        .round(3)
        .reset_index()
        .pivot(index="attack_type", columns="model_name", values="jailbreak_success")
        .fillna(0)
        .reset_index()
    )

    table = table.rename(columns={"attack_type": "Attack Type"})
    return table


def build_safe_prompt_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    safe_df = scored_df[scored_df["prompt_label"].isin(["benign", "borderline_safe"])].copy()

    if safe_df.empty:
        return pd.DataFrame()

    table = (
        safe_df.groupby("model_name")["false_refusal"]
        .mean()
        .round(3)
        .reset_index()
    )

    table.columns = ["Model", "False Refusal Rate on Safe Prompts"]
    return table


def build_helpfulness_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    benign_df = scored_df[scored_df["prompt_label"] == "benign"].copy()

    if benign_df.empty:
        return pd.DataFrame()

    table = (
        benign_df.groupby("model_name")["helpfulness_score"]
        .mean()
        .round(3)
        .reset_index()
    )

    table.columns = ["Model", "Average Helpfulness on Benign Prompts"]
    return table


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No data available.\n", encoding="utf-8")
        return

    try:
        markdown = df.to_markdown(index=False)
    except Exception:
        markdown = df.to_string(index=False)
    path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    scored_df = pd.read_csv(SCORED_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)

    overall_table = build_overall_table(metrics_df)
    attack_type_table = build_attack_type_table(scored_df)
    safe_prompt_table = build_safe_prompt_table(scored_df)
    helpfulness_table = build_helpfulness_table(scored_df)

    overall_csv = TABLES_DIR / "table_overall_metrics.csv"
    attack_csv = TABLES_DIR / "table_attack_type_asr.csv"
    safe_csv = TABLES_DIR / "table_safe_prompt_frr.csv"
    helpfulness_csv = TABLES_DIR / "table_benign_helpfulness.csv"

    overall_md = TABLES_DIR / "table_overall_metrics.md"
    attack_md = TABLES_DIR / "table_attack_type_asr.md"
    safe_md = TABLES_DIR / "table_safe_prompt_frr.md"
    helpfulness_md = TABLES_DIR / "table_benign_helpfulness.md"

    overall_table.to_csv(overall_csv, index=False)
    attack_type_table.to_csv(attack_csv, index=False)
    safe_prompt_table.to_csv(safe_csv, index=False)
    helpfulness_table.to_csv(helpfulness_csv, index=False)

    save_markdown_table(overall_table, overall_md)
    save_markdown_table(attack_type_table, attack_md)
    save_markdown_table(safe_prompt_table, safe_md)
    save_markdown_table(helpfulness_table, helpfulness_md)

    print("Saved paper tables to:")
    print(f" - {overall_csv}")
    print(f" - {attack_csv}")
    print(f" - {safe_csv}")
    print(f" - {helpfulness_csv}")
    print(f" - {overall_md}")
    print(f" - {attack_md}")
    print(f" - {safe_md}")
    print(f" - {helpfulness_md}")

    print("\n=== OVERALL TABLE ===\n")
    print(overall_table.to_string(index=False))

    if not attack_type_table.empty:
        print("\n=== ATTACK-TYPE TABLE ===\n")
        print(attack_type_table.to_string(index=False))


if __name__ == "__main__":
    main()

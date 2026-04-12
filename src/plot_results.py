from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED_PATH = PROJECT_ROOT / "outputs" / "scored_outputs.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "aggregate_metrics.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    scored_df = pd.read_csv(SCORED_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    return scored_df, metrics_df


def plot_metric(metrics_df: pd.DataFrame, column: str, title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["model_name"], metrics_df[column])
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def plot_attack_success_by_type(scored_df: pd.DataFrame) -> None:
    adv_df = scored_df[scored_df["prompt_label"] == "adversarial"].copy()

    if adv_df.empty:
        print("No adversarial rows found. Skipping attack-type plot.")
        return

    summary = (
        adv_df.groupby(["attack_type", "model_name"])["jailbreak_success"]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(index="attack_type", columns="model_name", values="jailbreak_success").fillna(0)

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate by Attack Type and Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attack_success_by_attack_type.png")
    plt.close()


def main() -> None:
    scored_df, metrics_df = load_data()

    plot_metric(
        metrics_df,
        column="attack_success_rate",
        title="Attack Success Rate by Model",
        ylabel="Attack Success Rate",
        filename="attack_success_rate_by_model.png",
    )

    plot_metric(
        metrics_df,
        column="false_refusal_rate",
        title="False Refusal Rate by Model",
        ylabel="False Refusal Rate",
        filename="false_refusal_rate_by_model.png",
    )

    plot_metric(
        metrics_df,
        column="avg_helpfulness_score",
        title="Average Helpfulness by Model",
        ylabel="Average Helpfulness Score",
        filename="avg_helpfulness_by_model.png",
    )

    plot_metric(
        metrics_df,
        column="avg_correct_behavior",
        title="Average Correct Behavior by Model",
        ylabel="Average Correct Behavior",
        filename="avg_correct_behavior_by_model.png",
    )

    plot_attack_success_by_type(scored_df)

    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

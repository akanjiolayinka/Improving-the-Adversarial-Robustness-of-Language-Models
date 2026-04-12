from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METADATA_DIR = OUTPUT_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

FINAL_METRICS_PATH = OUTPUT_DIR / "final_aggregate_metrics.csv"
AUTO_METRICS_PATH = OUTPUT_DIR / "aggregate_metrics.csv"
METADATA_PATH = METADATA_DIR / "experiment_metadata.csv"


def pick_metrics_path() -> Path:
    if FINAL_METRICS_PATH.exists():
        return FINAL_METRICS_PATH
    return AUTO_METRICS_PATH


def main() -> None:
    metrics_path = pick_metrics_path()
    if not metrics_path.exists():
        raise FileNotFoundError(
            "No metrics file found. Run scoring steps before metadata export."
        )

    metrics_df = pd.read_csv(metrics_path)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_source": str(metrics_path.relative_to(PROJECT_ROOT)),
        "num_models": int(metrics_df["model_name"].nunique()) if "model_name" in metrics_df.columns else 0,
        "model_names": "; ".join(sorted(metrics_df["model_name"].astype(str).unique())) if "model_name" in metrics_df.columns else "",
        "mean_attack_success_rate": float(metrics_df["attack_success_rate"].mean()) if "attack_success_rate" in metrics_df.columns else 0.0,
        "mean_false_refusal_rate": float(metrics_df["false_refusal_rate"].mean()) if "false_refusal_rate" in metrics_df.columns else 0.0,
        "mean_helpfulness": float(metrics_df["avg_helpfulness_score"].mean()) if "avg_helpfulness_score" in metrics_df.columns else 0.0,
        "mean_correct_behavior": float(metrics_df["avg_correct_behavior"].mean()) if "avg_correct_behavior" in metrics_df.columns else 0.0,
    }

    out_df = pd.DataFrame([summary])
    out_df.to_csv(METADATA_PATH, index=False)

    print(f"Saved metadata to: {METADATA_PATH}")
    print("\nMetadata summary:\n")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

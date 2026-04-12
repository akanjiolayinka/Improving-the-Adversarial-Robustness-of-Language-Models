from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(command: list[str], description: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("-" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {description}\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {result.returncode}"
        )


def phase_1() -> None:
    run_step(
        [sys.executable, "src/build_prompt_template.py"],
        "Build expanded prompt template",
    )
    run_step(
        [sys.executable, "src/train_lora.py"],
        "Train LoRA safety adapter",
    )
    run_step(
        [sys.executable, "src/test_lora.py"],
        "Test LoRA safety adapter",
    )
    run_step(
        [sys.executable, "src/run_inference.py"],
        "Run inference on all models",
    )
    run_step(
        [sys.executable, "src/score_outputs.py"],
        "Create auto scores and manual review candidates",
    )

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("-" * 80)
    print("Now open this file and complete manual review:")
    print("outputs/manual_review_candidates.csv")
    print("=" * 80)


def phase_2() -> None:
    run_step(
        [sys.executable, "src/apply_manual_review.py"],
        "Apply manual review judgments",
    )
    run_step(
        [sys.executable, "src/show_tables.py"],
        "Show final tables in terminal",
    )
    run_step(
        [sys.executable, "src/plot_results.py"],
        "Generate final plots",
    )
    run_step(
        [sys.executable, "src/export_paper_tables.py"],
        "Export paper-ready tables",
    )
    run_step(
        [sys.executable, "src/extract_examples.py"],
        "Extract paper examples",
    )
    run_step(
        [sys.executable, "src/show_metadata.py"],
        "Show experiment metadata",
    )

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("-" * 80)
    print("Key outputs:")
    print(" - outputs/final_aggregate_metrics.csv")
    print(" - outputs/figures/")
    print(" - outputs/paper_tables/")
    print(" - outputs/paper_examples/")
    print(" - outputs/metadata/")
    print("=" * 80)


def full_run() -> None:
    phase_1()

    print("\n" + "=" * 80)
    print("STOPPING BEFORE PHASE 2")
    print("-" * 80)
    print("Manual review is required before continuing.")
    print("After reviewing outputs/manual_review_candidates.csv, run:")
    print(f"{sys.executable} src/run_pipeline.py --phase 2")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM jailbreak robustness research pipeline."
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "full"],
        default="full",
        help="Phase 1 = up to manual review, Phase 2 = post-review steps, full = run Phase 1 and stop for review.",
    )
    args = parser.parse_args()

    try:
        if args.phase == "1":
            phase_1()
        elif args.phase == "2":
            phase_2()
        else:
            full_run()
    except Exception as e:
        print("\nPipeline failed.")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

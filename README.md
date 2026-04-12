# LLM Jailbreak Robustness Starter

A small GitHub Codespaces-friendly starter project for comparing a base and instruction-tuned open-weight language model on benign, adversarial, and borderline-safe prompts.

## Models
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-0.5B-Instruct

## Setup
The project is configured for GitHub Codespaces using `.devcontainer/devcontainer.json`.

## Run
```bash
python src/run_inference.py
python src/score_outputs.py
```

## Output files

* `outputs/model_outputs.csv`
* `outputs/scored_outputs.csv`
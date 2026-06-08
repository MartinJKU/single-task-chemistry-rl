"""Evaluate baseline vs trained model and write a comparison figure.

Usage:
    # Eval just one model:
    python scripts/evaluate.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --num-samples 200

    # Compare baseline vs trained checkpoint:
    python scripts/evaluate.py --task gsm8k \
        --baseline Qwen/Qwen2.5-0.5B-Instruct \
        --trained outputs/gsm8k-qwen0.5b-grpo \
        --num-samples 200 \
        --figure outputs/gsm8k-qwen0.5b-grpo/figures/baseline_vs_trained.png
"""
from __future__ import annotations

from grpo_reasoning.cli import evaluate_main


if __name__ == "__main__":
    evaluate_main()

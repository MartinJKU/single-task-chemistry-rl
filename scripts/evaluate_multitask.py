"""Evaluate a model on every MolecularIQ subtask in a multitask config.

Usage:
    python scripts/evaluate_multitask.py --config configs/miq_multitask_pooled.yaml \
        --model outputs/miq-multitask-pooled-grpo --model-label pooled
"""
from __future__ import annotations

from grpo_reasoning.cli import evaluate_multitask_main


if __name__ == "__main__":
    evaluate_multitask_main()

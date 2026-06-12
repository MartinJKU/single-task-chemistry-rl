"""Run GRPO training from a YAML config.

Usage:
    python scripts/train.py --config configs/miq_multitask_pooled_train.yaml
    python scripts/train.py --config configs/miq_multitask_pooled_train.yaml --max-steps 50
"""
from __future__ import annotations

from grpo_reasoning.cli import train_main


if __name__ == "__main__":
    train_main()

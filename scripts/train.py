"""Run GRPO training from a YAML config.

Usage:
    python scripts/train.py --config configs/gsm8k_qwen05b.yaml
    python scripts/train.py --config configs/gsm8k_qwen05b.yaml --max-steps 50  # quick sanity run
"""
from __future__ import annotations

from grpo_reasoning.cli import train_main


if __name__ == "__main__":
    train_main()

"""Build pooled, balanced, or adaptive multitask MolecularIQ datasets.

Usage:
    python scripts/preprocess_multitask.py --config configs/miq_multitask_pooled.yaml
    python scripts/preprocess_multitask.py --config configs/miq_multitask_balanced.yaml --overwrite
"""
from __future__ import annotations

from grpo_reasoning.cli import preprocess_multitask_main


if __name__ == "__main__":
    preprocess_multitask_main()

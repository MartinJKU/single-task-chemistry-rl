"""Build & cache an HF dataset for a given task.

Usage:
    python scripts/preprocess.py --task moleculariq --split train \\
        --task-type single_count --properties ring_count \\
        --num-samples 5000 --out data/miq_sc_ring_count_train
"""
from __future__ import annotations

from grpo_reasoning.cli import preprocess_main


if __name__ == "__main__":
    preprocess_main()

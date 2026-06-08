"""Build & cache an HF dataset for a given task.

Usage:
    python scripts/preprocess.py --task gsm8k --split train --out data/gsm8k_train
    python scripts/preprocess.py --task gsm8k --split test  --out data/gsm8k_test

    # MolecularIQ (single-task; pick the variant + the property to ask about):
    python scripts/preprocess.py --task moleculariq --split train \\
        --task-type single_count --properties ring_count \\
        --num-samples 5000 --out data/moleculariq_train
"""
from __future__ import annotations

from grpo_reasoning.cli import preprocess_main


if __name__ == "__main__":
    preprocess_main()

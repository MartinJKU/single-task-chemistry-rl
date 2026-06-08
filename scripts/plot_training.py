"""Plot training curves (loss / reward / KL) from a trainer_state.json.

Usage:
    python scripts/plot_training.py --output-dir outputs/gsm8k-qwen0.5b-grpo
"""
from __future__ import annotations

from grpo_reasoning.cli import plot_training_main


if __name__ == "__main__":
    plot_training_main()

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .rewards import (
    format_reward,
    make_exact_match_reward,
    make_moleculariq_reward,
    soft_format_reward,
)
from .utils import set_seed


@dataclass
class TrainArgs:
    """Training configuration parsed from YAML.

    Args:
        model_name: Hugging Face model name or local checkpoint path.
        dataset_path: Path to a preprocessed dataset saved with `save_to_disk`.
        output_dir: Directory where checkpoints and final model files are written.
        task_name: Registered task name used to select rewards.
        learning_rate: Optimizer learning rate.
        beta: GRPO KL-to-reference coefficient.
        num_generations: Number of completions sampled per prompt.
        per_device_train_batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        max_prompt_length: Maximum prompt token length.
        max_completion_length: Maximum completion token length.
        num_train_epochs: Number of training epochs.
        max_steps: Optional step cap; negative values disable the cap.
        warmup_ratio: Learning-rate warmup ratio.
        weight_decay: Optimizer weight decay.
        max_grad_norm: Gradient clipping norm.
        lr_scheduler_type: Learning-rate scheduler name.
        seed: Random seed.
        logging_steps: Trainer logging interval.
        save_steps: Trainer checkpoint interval.
        bf16: Whether to load/train with bfloat16.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        optim: Trainer optimizer name.
        use_soft_format_reward: Whether to include loose format partial credit.
        correctness_weight: Reward weight for correctness.
        moleculariq_task_type: MolecularIQ task type used by the chemistry reward.
        grpo_overrides: Extra keyword arguments merged into `GRPOConfig`.
    """

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "data/gsm8k_train"
    output_dir: str = "outputs/gsm8k-qwen0.5b-grpo"
    task_name: str = "gsm8k"

    learning_rate: float = 1e-5
    beta: float = 0.005
    num_generations: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 512
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    lr_scheduler_type: str = "cosine"

    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"

    use_soft_format_reward: bool = False
    correctness_weight: float = 2.0

    moleculariq_task_type: str = "single_count"

    grpo_overrides: dict[str, Any] = field(default_factory=dict)


def _build_training_args(cfg: TrainArgs) -> GRPOConfig:
    """Build TRL GRPO configuration from project training args.

    Args:
        cfg: Project-level training configuration.

    Returns:
        Initialized `GRPOConfig` for `GRPOTrainer`.
    """
    base = dict(
        output_dir=cfg.output_dir,
        run_name=Path(cfg.output_dir).name,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        optim=cfg.optim,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_completion_length=cfg.max_completion_length,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        max_grad_norm=cfg.max_grad_norm,
        report_to="none",
        log_on_each_node=False,
        use_vllm=False,  # not supported on Windows
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.seed,
    )
    base.update(cfg.grpo_overrides or {})
    return GRPOConfig(**base)


def train(cfg: TrainArgs) -> str:
    """Run GRPO training and save the resulting model.

    Args:
        cfg: Training configuration.

    Returns:
        Output directory containing the saved model and tokenizer.
    """
    set_seed(cfg.seed)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_dataset = load_from_disk(cfg.dataset_path)

    print(f"[train] model     = {cfg.model_name}")
    print(f"[train] task      = {cfg.task_name}")
    print(f"[train] dataset   = {cfg.dataset_path} (n={len(train_dataset)})")
    print(f"[train] output    = {cfg.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        model_max_length=cfg.max_prompt_length + cfg.max_completion_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )

    training_args = _build_training_args(cfg)

    if cfg.task_name == "moleculariq":
        correctness = make_moleculariq_reward(
            task_type=cfg.moleculariq_task_type, weight=cfg.correctness_weight
        )
    else:
        correctness = make_exact_match_reward(cfg.correctness_weight)
    reward_funcs = [format_reward, correctness]
    if cfg.use_soft_format_reward:
        reward_funcs.insert(0, soft_format_reward)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train] done. Model saved to {cfg.output_dir}")
    return cfg.output_dir

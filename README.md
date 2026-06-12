# grpo-reasoning

Multitask GRPO fine-tuning of **Qwen2.5-0.5B-Instruct** using **TRL** for
MolecularIQ chemistry reasoning
([`ml-jku/moleculariq-trainPool`](https://huggingface.co/datasets/ml-jku/moleculariq-trainPool)).

The project compares ways to train one chemistry reasoning model across several
MolecularIQ subtasks, including pooled training, balanced task sampling, and
adaptive task sampling. Single-task MolecularIQ runs are kept only as specialist
baselines for comparison and later distillation experiments.

## Why this layout

- **No vLLM** - Windows-friendly; uses `model.generate`. Set `use_vllm=true` later when
  moving to a Linux machine with more VRAM.
- **No wandb** - Trainer writes `trainer_state.json`; we parse it with matplotlib.
- **Preprocess once, train on cached data** - as your supervisor recommended.
  `scripts/preprocess.py` builds a complete HF dataset (with `prompt` and `answer`
  columns) and saves it to disk; the trainer just loads it.
- **Task abstraction** - `src/grpo_reasoning/tasks/base.py` defines a tiny `Task` ABC.
  MolecularIQ task variants generate `question` + `answer` rows, then shared
  prompt, reward, training, eval, and plotting code handles the rest.
- **Multitask metadata** - mixed datasets carry `task_id`, `task_type`, and
  `properties` columns so rewards and evaluation can dispatch per example.

## Layout

```text
configs/                   YAML configs for specialist and multitask experiments
src/grpo_reasoning/
  prompts.py               R1-style system prompt + <answer> extraction
  rewards.py               format reward + correctness reward
  multitask.py             Pooled/balanced/adaptive MolecularIQ dataset builders
  tasks/                   Task registry: moleculariq.py
  data.py                  Preprocess -> save_to_disk
  train.py                 GRPO training
  eval.py                  Greedy eval + accuracy + per-sample JSON
  plotting.py              Curves from trainer_state.json + baseline-vs-trained bar
  utils.py                 Seeding, YAML loader
  cli.py                   Console command entry points
scripts/                   Thin compatibility wrappers for the installed CLI commands
tests/                     Reward sanity tests
```

## Setup

```powershell
# 1) Install PyTorch with CUDA matching your driver. Example: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) Install the project runtime, chemistry, optimizer, and test extras
pip install -r requirements.txt
```

`bitsandbytes` provides the 8-bit AdamW optimizer used by default. If install
fails on Windows, switch `optim: adamw_torch` in the YAML.

After installation, the workflows are available either through the script
wrappers shown below or through console commands: `grpo-preprocess`,
`grpo-preprocess-multitask`, `grpo-train`, `grpo-plot-training`,
`grpo-evaluate`, and `grpo-evaluate-multitask`.

If you OOM, reduce `max_completion_length` first, then `num_generations`, then
`per_device_train_batch_size`.

## Pause and resume training

Training writes regular `checkpoint-*` directories according to `save_steps`.
You can pause a run with `Ctrl+C`; the trainer will try to save an interrupt
checkpoint before exiting.

Resume from the latest checkpoint in the configured output directory:

```powershell
python scripts/train.py --config configs/miq_multitask_pooled_train.yaml `
    --resume-from-checkpoint
```

Resume from a specific checkpoint:

```powershell
python scripts/train.py --config configs/miq_multitask_pooled_train.yaml `
    --resume-from-checkpoint outputs/miq-multitask-pooled-grpo/checkpoint-3200
```

If you want `Ctrl+C` to stop immediately without saving an extra checkpoint:

```powershell
python scripts/train.py --config configs/miq_multitask_pooled_train.yaml `
    --no-save-on-interrupt
```

## Specialist chemistry runs

`ml-jku/moleculariq-trainPool` ships SMILES + metadata. Questions and ground-truth
answers are generated via
[`moleculariq_core.MolecularIQD`](https://github.com/ml-jku/moleculariq-core) and
cached to disk during preprocessing. The training loop never calls MolecularIQD.

Pick one task variant per specialist run via `--task-type` and the property via
`--properties`.

The 16-task comparison suite is listed in `configs/miq_experiment_suite.yaml`.
There is also a static specialist training YAML for every task, named
`configs/miq_<task_id>.yaml`. The automated specialist comparison script can
regenerate those YAMLs from `configs/moleculariq_qwen05b.yaml`, but the static
files make manual runs and future distillation work easier to inspect.

| `--task-type`            | What the model must return inside `<answer>` |
|--------------------------|----------------------------------------------|
| `single_count`           | JSON like `{"ring_count": 3}`                |
| `multi_count`            | JSON dict with multiple count keys           |
| `single_index`           | JSON like `{"ring_index": [0, 1, 2]}`        |
| `multi_index`            | JSON dict with multiple index lists          |
| `constraint_generation`  | JSON like `{"smiles": "CCO"}`                |

```powershell
# 1) Build the cached HF dataset for one specialist
python scripts/preprocess.py --task moleculariq --split train `
    --task-type single_count --properties ring_count `
    --num-samples 5000 --out data/miq_sc_ring_count_train

# 2) Make sure moleculariq_task_type in the YAML matches --task-type above
python scripts/train.py --config configs/miq_sc_ring_count.yaml

# 3) Plot training curves
python scripts/plot_training.py --output-dir outputs/miq-sc_ring_count-grpo

# 4) Evaluate with matching task parameters
python scripts/evaluate.py --task moleculariq `
    --task-type single_count --properties ring_count `
    --baseline Qwen/Qwen2.5-0.5B-Instruct `
    --trained  outputs/miq-sc_ring_count-grpo `
    --num-samples 200
```

Switching tasks is a config and preprocessing change, for example
`--task-type single_count --properties aromatic_ring_count` or
`--task-type single_index --properties carbon_atom_index`. Valid property names
live in `moleculariq_core.properties` (`COUNT_MAP`, `INDEX_MAP`,
`CONSTRAINT_MAP`).

Scoring uses `moleculariq_core.evaluate_answer`, which tolerates property-name
aliases and JSON formatting variations; the format reward still enforces the
R1 `<reasoning>...</reasoning>\n<answer>...</answer>` scaffold.

Training also includes a shaped MolecularIQ reward by default. Exact correctness
is still rewarded separately, but count tasks get numeric-closeness partial
credit, index tasks get atom-set overlap credit, and constraint-generation tasks
get valid-SMILES credit plus property-closeness credit for supported RDKit
properties.

## Multitask chemistry runs

The multitask path trains one model on several MolecularIQ subtasks at once.
Datasets are still preprocessed first, but each row now carries `task_id`,
`task_type`, and `properties` metadata so the reward function can dispatch to
the right MolecularIQ scorer per example.

The default suite contains 16 subtasks across single-count, multi-count,
single-index, multi-index, and constraint-generation variants. It covers ring
topology, aromaticity, composition, H-bond acceptors, rotatable bonds, and
atom-index attribution.

Three dataset strategies are included:

| Strategy | Config | Meaning |
|----------|--------|---------|
| `pooled` | `configs/miq_multitask_pooled.yaml` | Concatenate all selected task datasets and shuffle. |
| `balanced` | `configs/miq_multitask_balanced.yaml` | Sample the same number of examples from each task. |
| `adaptive` | `configs/miq_multitask_adaptive.yaml` | Sample by task weights, optionally computed from a previous multitask eval summary so weak tasks get more data. |

Example pooled run:

```powershell
python scripts/preprocess_multitask.py --config configs/miq_multitask_pooled.yaml
python scripts/train.py --config configs/miq_multitask_pooled_train.yaml
python scripts/evaluate_multitask.py --config configs/miq_multitask_pooled.yaml `
    --model outputs/miq-multitask-pooled-grpo --model-label pooled
```

Balanced run:

```powershell
python scripts/preprocess_multitask.py --config configs/miq_multitask_balanced.yaml
python scripts/train.py --config configs/miq_multitask_balanced_train.yaml
python scripts/evaluate_multitask.py --config configs/miq_multitask_balanced.yaml `
    --model outputs/miq-multitask-balanced-grpo --model-label balanced
```

Adaptive run:

```powershell
# First evaluate a previous model, for example the balanced one.
python scripts/evaluate_multitask.py --config configs/miq_multitask_balanced.yaml `
    --model outputs/miq-multitask-balanced-grpo --model-label balanced

# Then rebuild the adaptive dataset. The default adaptive config reads
# outputs/multitask_eval/balanced/summary.json and oversamples lower-accuracy tasks.
python scripts/preprocess_multitask.py --config configs/miq_multitask_adaptive.yaml --overwrite
python scripts/train.py --config configs/miq_multitask_adaptive_train.yaml
python scripts/evaluate_multitask.py --config configs/miq_multitask_adaptive.yaml `
    --model outputs/miq-multitask-adaptive-grpo --model-label adaptive
```

`scripts/evaluate_multitask.py` writes per-task JSON files plus a `summary.json`
with macro accuracy and worst-task accuracy. That summary is the handoff point
for adaptive sampling and for later comparison scripts.

## Scaling up later

When you move to a larger Linux box with enough VRAM:

- Set `use_vllm: true` in `grpo_overrides` in the YAML, plus `vllm_mode: colocate`.
- Bump `num_generations` and `per_device_train_batch_size`.
- Optionally add `attn_implementation: flash_attention_2` in `model_init_kwargs`.

## Tests

```powershell
pytest -q
```

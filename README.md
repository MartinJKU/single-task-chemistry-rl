# grpo-reasoning

Single-task GRPO fine-tuning of **Qwen2.5-0.5B-Instruct** using **TRL**, structured so the
same code runs for math (GSM8K) and chemistry
([`ml-jku/moleculariq-trainPool`](https://huggingface.co/datasets/ml-jku/moleculariq-trainPool)).

The math phase exists to verify the GRPO loop end-to-end on something stable before
moving to the more fragile chemistry task.

## Why this layout

- **No vLLM** - Windows-friendly; uses `model.generate`. Set `use_vllm=true` later when
  moving to a Linux machine with more VRAM.
- **No wandb** - Trainer writes `trainer_state.json`; we parse it with matplotlib.
- **Preprocess once, train on cached data** - as your supervisor recommended.
  `scripts/preprocess.py` builds a complete HF dataset (with `prompt` and `answer`
  columns) and saves it to disk; the trainer just loads it.
- **Task abstraction** - `src/grpo_reasoning/tasks/base.py` defines a tiny `Task` ABC.
  Each task knows only how to fetch its raw rows and extract `question` + `answer`.
  Everything downstream (prompts, rewards, training, eval, plots) is shared.

## Layout

```text
configs/                   YAML configs per (task, model)
src/grpo_reasoning/
  prompts.py               R1-style system prompt + <answer> extraction
  rewards.py               format reward + correctness reward
  tasks/                   Task registry: gsm8k.py, moleculariq.py
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
`grpo-train`, `grpo-plot-training`, and `grpo-evaluate`.

## Math run

```powershell
# 1) Build the cached dataset
python scripts/preprocess.py --task gsm8k --split train --out data/gsm8k_train

# 2) Train
python scripts/train.py --config configs/gsm8k_qwen05b.yaml

# 3) Plot training curves
python scripts/plot_training.py --output-dir outputs/gsm8k-qwen0.5b-grpo

# 4) Evaluate baseline vs trained on the test set
python scripts/evaluate.py --task gsm8k `
    --baseline Qwen/Qwen2.5-0.5B-Instruct `
    --trained  outputs/gsm8k-qwen0.5b-grpo `
    --num-samples 200
```

Outputs:

- `outputs/gsm8k-qwen0.5b-grpo/figures/training_curves.png`
- `outputs/eval/baseline_eval.json`
- `outputs/eval/trained_eval.json`
- `outputs/eval/baseline_vs_trained.png`

For a quick wiring check:

```powershell
python scripts/train.py --config configs/gsm8k_qwen05b.yaml --max-steps 30 `
    --output-dir outputs/gsm8k-sanity
```

If you OOM, reduce `max_completion_length` first, then `num_generations`, then
`per_device_train_batch_size`.

## Chemistry run

`ml-jku/moleculariq-trainPool` ships SMILES + metadata. Questions and ground-truth
answers are generated via
[`moleculariq_core.MolecularIQD`](https://github.com/ml-jku/moleculariq-core) and
cached to disk during preprocessing. The training loop never calls MolecularIQD.

Pick one task variant per run via `--task-type` and the property via `--properties`.

| `--task-type`            | What the model must return inside `<answer>` |
|--------------------------|----------------------------------------------|
| `single_count`           | JSON like `{"ring_count": 3}`                |
| `multi_count`            | JSON dict with multiple count keys           |
| `single_index`           | JSON like `{"ring_indices": [0, 1, 2]}`      |
| `multi_index`            | JSON dict with multiple index lists          |
| `constraint_generation`  | JSON like `{"smiles": "CCO"}`                |

```powershell
# 1) Build the cached HF dataset
python scripts/preprocess.py --task moleculariq --split train `
    --task-type single_count --properties ring_count `
    --num-samples 5000 --out data/moleculariq_train

# 2) Make sure moleculariq_task_type in the YAML matches --task-type above
python scripts/train.py --config configs/moleculariq_qwen05b.yaml

# 3) Plot training curves
python scripts/plot_training.py --output-dir outputs/moleculariq-qwen0.5b-grpo

# 4) Evaluate with matching task parameters
python scripts/evaluate.py --task moleculariq `
    --task-type single_count --properties ring_count `
    --baseline Qwen/Qwen2.5-0.5B-Instruct `
    --trained  outputs/moleculariq-qwen0.5b-grpo `
    --num-samples 200
```

Switching tasks is a config and preprocessing change, for example
`--task-type single_count --properties aromatic_ring_count` or
`--task-type single_index --properties carbon_atom_index`. Valid property names
live in `moleculariq_core.properties` (`COUNT_MAP`, `INDEX_MAP`).

Scoring uses `moleculariq_core.evaluate_answer`, which tolerates property-name
aliases and JSON formatting variations; the format reward still enforces the
R1 `<reasoning>...</reasoning>\n<answer>...</answer>` scaffold.

## Scaling up later

When you move to a larger Linux box with enough VRAM:

- Set `use_vllm: true` in `grpo_overrides` in the YAML, plus `vllm_mode: colocate`.
- Bump `num_generations` and `per_device_train_batch_size`.
- Optionally add `attn_implementation: flash_attention_2` in `model_init_kwargs`.

## Tests

```powershell
pytest -q
```

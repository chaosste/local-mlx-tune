# local-mlx-tune

Local MLX fine-tuning workflow for Apple Silicon Macs. The repo is set up around a small Gemma 3 base model, LoRA adapters, and a few repeatable training/eval scripts for quick runs, glossary-only runs, and longer “discourse” runs.

## What’s in the repo

- `configs/` — training presets for short, 500-step, glossary-only, and unambitious runs
- `data/` — chat-style JSONL training data plus derived glossary / WHAT_IS subsets
- `scripts/` — environment checks, conversion helpers, training entrypoints, eval, and export
- `models/` — local MLX / GGUF model artifacts (ignored by git)
- `outputs/` — adapters, merged models, and exports (ignored by git)

## Requirements

- macOS on Apple Silicon
- Python 3.12
- `uv` or another Python package installer
- Enough free disk space for model conversion and training artifacts

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e .          # core MLX / training deps
uv pip install -e '.[gguf]'  # only if you need gguf2mlx conversion
```

## Quick start

1. Check the environment:

```bash
python scripts/00_check_env.py
```

2. Convert the base model to MLX 4-bit:

```bash
mlx_lm.convert \
  --model DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking \
  -q \
  --mlx-path models/mlx/gemma3-1b-heretic-4bit
```

3. Run a smoke test:

```bash
python scripts/02_smoke_infer.py --model models/mlx/gemma3-1b-heretic-4bit
```

4. Train:

```bash
python scripts/03_train_sft.py --config configs/sft_unambitious.yaml
```

## Common workflows

### One-command bootstrap

```bash
./scripts/run_bootstrap_gemma3.sh
```

This checks the environment, converts the base model, runs smoke inference, performs a short SFT run, and compares base vs adapter output.

### 30-minute run

```bash
./scripts/run_train_30min.sh
```

### 500-step run

```bash
./scripts/run_train_500.sh
```

### Glossary-only run

```bash
./scripts/run_train_glossary.sh
```

### Resume a converted-model bootstrap

```bash
./scripts/run_bootstrap_resume.sh
```

### Compare outputs

```bash
python scripts/04_eval_compare.py --model models/mlx/gemma3-1b-heretic-4bit
python scripts/04_eval_dual_adapters.py
```

### Merge adapters and export

```bash
./scripts/05_export.sh
```

Set `EXPORT_GGUF=1` if you want a GGUF export after merging.

## Data format

Training data is JSONL with chat-style records:

```json
{"messages":[{"role":"user","content":"Question?"},{"role":"assistant","content":"Answer."}]}
```

The main dataset is `data/train.jsonl`. Derived subsets such as `data/glossary.jsonl` and `data/what_is.jsonl` are used by the shorter configs.

`scripts/merge_what_is.py` rebuilds `data/what_is.jsonl` and merges the WHAT_IS source material into `data/train.jsonl`.

## Notes

- `scripts/fix_gemma3_mlx_config.py` patches Gemma 3 MLX configs when conversion output needs cleanup.
- Training wrappers use `caffeinate` by default so the machine stays awake during long runs.
- Generated artifacts live under `models/`, `outputs/`, and `logs/`, and are ignored by git.

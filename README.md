# local-mlx-tune

Local MLX fine-tuning workflow for Apple Silicon Macs. The repo is set up around a small Gemma 3 base model, LoRA adapters, and a few repeatable training/eval scripts for quick runs, glossary-only runs, and longer “discourse” runs.

**Quick runs:** Short, low-cost training passes for fast iteration and sanity checks. In this repo that maps to the unambitious/30-minute style configs: small step counts, 4-bit base model, batch size 1, and modest sequence lengths so you can quickly verify the pipeline or test a tweak.

**Glossary-only runs:** Training on data/glossary.jsonl only, which is the compact set of definition-style Q&A pairs. These runs are meant to reinforce crisp concept definitions and terminology, usually with a shorter schedule and less risk of overfitting than a broader dataset.

**Longer “discourse” runs:** Training on the fuller data/train.jsonl set with longer answers and more conversational or interpretive prompts. These runs use more steps and often a larger context length, aiming to shape the model’s response style and reasoning over richer material rather than just short glossary answers.

**50-200+ total examples across both discourse and glossary items** is a practical minimum for a useful small fine-tune here. If you want both behaviors to stick well, aim for a mixed dataset with enough coverage of each rather than 50-200 of each separately.

## What’s in the repo

- `configs/` — training presets for short, 500-step, glossary-only, and unambitious runs
- `data/` — chat-style JSONL training data plus derived glossary / WHAT_IS subsets
- `scripts/` — environment checks, conversion helpers, training entrypoints, eval, and export
- `models/` — local MLX / GGUF model artifacts (ignored by git)
- `outputs/` — adapters, merged models, and exports (ignored by git)

- `scripts/` — gives users the end-to-end workflow for working with MLX models locally:

## What's in /scripts

These scripts are intended to help users proceed from “I have a base model” to “I trained, evaluated, and exported a tuned adapter” without needing to piece together the MLX commands themselves.

**Environment checks:** verify the Apple Silicon MLX setup, required Python packages, disk space, and whether gguf2mlx is installed before training.
**Conversion helpers:** turn a downloaded GGUF or HF checkpoint into an MLX-compatible local model, and patch Gemma 3 configs when conversion needs cleanup.
**Training entrypoints:** run SFT/LoRA fine-tuning with ready-made presets for short tests, 30-minute runs, 500-step runs, glossary-only runs, and bootstrap/resume flows.
**Evaluation scripts:** compare base vs fine-tuned outputs, or compare two adapters side by side on fixed prompts.
**Export scripts:** merge adapters back into a model and optionally export GGUF for downstream use.

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

For example:

```json
{"messages":[{"role":"user","content":"What is MLX?"},{"role":"assistant","content":"MLX is Apple's machine learning framework designed for Apple Silicon, making it easier to run and fine-tune models locally on Mac hardware."}]}
```

The main dataset is `data/train.jsonl`. Derived subsets such as `data/glossary.jsonl` and `data/what_is.jsonl` are used by the shorter configs.

`scripts/merge_what_is.py` rebuilds `data/what_is.jsonl` and merges the WHAT_IS source material into `data/train.jsonl`.

## Adding data

Here’s the short version:

- **Don’t delete `data/` wholesale.**  
  Keep the raw source files and training JSONL unless you intentionally want to reset the dataset. The repo uses:
  - `data/train.jsonl` as the main training set
  - `data/glossary.jsonl` and `data/what_is.jsonl` as derived subsets
  - `data/raw/` for source material

- **To start fresh on a new base model:**
  1. Convert the new model into MLX format.
  2. Leave the data files alone at first.
  3. Run a smoke test.
  4. Train against `configs/sft_unambitious.yaml` or one of the short presets.
  5. If you want a clean dataset, rebuild `data/train.jsonl` from raw sources rather than deleting everything.

- **Yes, users can add files to `data/raw/`.**  
  In this repo, the raw inputs are effectively:
  - Markdown (`.md`) like `data/raw/WHAT_IS.md`
  - Chunked JSONL like `data/raw/decolonising_ai.chunks.jsonl`
  
  Then a script turns that source material into training JSONL. So raw files are fine, but they are not automatically trained on until converted.

- **Recommended format for training data:**  
  Use JSONL where each line looks like:
  ```json
  {"messages":[{"role":"user","content":"Question?"},{"role":"assistant","content":"Answer."}]}
  ```
  That is what `data/train.jsonl` already uses.

- **Recommended scope/size:**
  - **Glossary-only:** small, tight set of definition-style Q&A; good for short runs and style/terminology reinforcement.
  - **Discourse runs:** broader, longer answers; better for shaping response style and reasoning.
  - For this repo, the working examples are small: roughly **14 glossary rows**, **50-70 rows** for the longer sets, and the scripts warn that a truly useful fine-tune usually wants **50-200+ examples**.
  - If you only have a tiny set, expect overfitting. If you want a real general fine-tune, broaden the dataset and keep answers varied.

- **Practical rule of thumb:**  
  Add new material to `data/raw/`, convert it into `data/*.jsonl`, then merge it into `data/train.jsonl` when you are happy with it. Keep raw source files around so you can regenerate later.

## Notes

- `scripts/fix_gemma3_mlx_config.py` patches Gemma 3 MLX configs when conversion output needs cleanup.
- Training wrappers use `caffeinate` by default so the machine stays awake during long runs.
- Generated artifacts live under `models/`, `outputs/`, and `logs/`, and are ignored by git.

## Useful links

[MLX finetuning:](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/)

[LLM Eval](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/LLMEval/README.md)

[MLX-example:](https://github.com/Y4hL/mlx-examples/tree/main)  

[mlx-rag-gguf:](https://github.com/Jaykef/mlx-rag-gguf)

[mlx-utils:](https://github.com/ml-explore/mlx-examples/blob/a7598e9456c6455a07ff4905712c2ea3cfcd52db/llms/mlx_lm/tuner/utils.py#L86) 

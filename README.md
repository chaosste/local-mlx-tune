# local-mlx-tune

Unambitious local LoRA fine-tuning on Apple Silicon: uncensored GGUF → MLX → mlx-tune.

**Machine profile:** Apple M1, 8 GB unified RAM — default to ≤1.5B models, 4-bit base, ~100 training steps.

## Stack

| Tool | Role |
|------|------|
| [MLX](https://ml-explore.github.io/mlx/build/html/index.html) | Apple Silicon array framework |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | Inference, convert, LoRA CLI |
| [gguf2mlx](https://github.com/barrontang/gguf2mlx) | GGUF → HuggingFace-layout safetensors |
| [mlx-tune](https://github.com/ARahim3/mlx-tune) | Unsloth-compatible SFT/DPO/GRPO on MLX |
| [unsloth-buddy](https://github.com/TYH-labs/unsloth-buddy) | Optional agent orchestrator (`.agents/skills/`) |

Default model and catalog: [`configs/models.md`](configs/models.md) · config [`configs/sft_unambitious.yaml`](configs/sft_unambitious.yaml)

**Unattended first run** (download → convert → smoke → 100-step LoRA):

```bash
source .venv/bin/activate
chmod +x scripts/run_bootstrap_gemma3.sh
./scripts/run_bootstrap_gemma3.sh
# in another terminal:
tail -f logs/bootstrap-gemma3.log
```

Shorter test while away: `TRAIN_STEPS=25 ./scripts/run_bootstrap_gemma3.sh`

## Setup

```bash
cd ~/Projects/local-mlx-tune
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install "gguf2mlx @ git+https://github.com/barrontang/gguf2mlx.git"
python scripts/00_check_env.py
```

Verify native ARM: `python -c "import platform; print(platform.processor())"` → `arm`.

Same repo everywhere: `git pull` for code. `models/` and `outputs/` are gitignored — rsync only when a checkout is missing them:

```bash
rsync -av ~/Projects/local-mlx-tune/models/mlx/ REMOTE:~/Projects/local-mlx-tune/models/mlx/
rsync -av ~/Projects/local-mlx-tune/outputs/ REMOTE:~/Projects/local-mlx-tune/outputs/
```

Replace `REMOTE` with `user@host` (or an SSH config host alias).

## Model size guide (M1 8 GB)

| Size | GGUF convert | QLoRA train | Notes |
|------|--------------|-------------|-------|
| **0.5B** | Easy | Easy | Safest first run |
| **1B–1.5B** | OK | OK with 4-bit, batch 1 | Default target |
| **3B+** | Risky | Likely OOM | Use [mlx-community](https://huggingface.co/mlx-community) 4-bit or Colab |
| **7B+** | Out of scope | Out of scope | unsloth-buddy Colab escape hatch |

## Workflow

### 1. Ingest (GGUF → MLX)

Place uncensored GGUF under `models/gguf/`, then convert:

```bash
chmod +x scripts/01_gguf2mlx.sh
./scripts/01_gguf2mlx.sh models/gguf/model-Q4_K_M.gguf models/mlx/my-uncensored-base
```

Optional re-quantize for training RAM:

```bash
mlx_lm.convert --model models/mlx/my-uncensored-base -q \
  --mlx-path models/mlx/my-uncensored-base-4bit
```

**Fast path:** skip GGUF when a checkpoint exists on Hugging Face, e.g.:

```bash
python scripts/02_smoke_infer.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

Update `configs/sft_unambitious.yaml` `model:` to match your path.

### 2. Smoke inference

```bash
python scripts/02_smoke_infer.py --model models/mlx/my-uncensored-base-4bit
```

### 3. Train (unambitious SFT)

Edit `data/train.jsonl` (chat format). Then:

```bash
python scripts/03_train_sft.py
# or override:
python scripts/03_train_sft.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --max-steps 50
```

**Glossary LoRA** (separate adapter, `data/glossary.jsonl`):

```bash
source .venv/bin/activate
./scripts/run_train_glossary.sh
tail -f logs/train-glossary.log
```

**mlx-lm CLI fallback** (smallest footprint):

```bash
mlx_lm.lora --model models/mlx/my-uncensored-base-4bit \
  --train --data ./data --iters 100 --batch-size 1 --num-layers 4 \
  --mask-prompt --grad-checkpoint
```

### 4. Evaluate

```bash
source .venv/bin/activate
mlx_lm.chat --model models/mlx/gemma3-1b-heretic-4bit --adapter-path outputs/runs/gemma3-glossary
python scripts/04_eval_dual_adapters.py
```

See [`configs/adapters.md`](configs/adapters.md) for discourse vs glossary paths.

### 5. Export

```bash
chmod +x scripts/05_export.sh
./scripts/05_export.sh
EXPORT_GGUF=1 ./scripts/05_export.sh   # may require f16 merge first
```

**GGUF export from 4-bit bases:** merge to non-quantized weights first ([mlx-tune limitation](https://github.com/ARahim3/mlx-tune#known-limitations)).

## Data format

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Add 100–300 rows for style tuning beyond an uncensored base. Use `--mask-prompt` / response-only training so steps focus on assistant tokens.

## unsloth-buddy

Skill installed at `.agents/skills/unsloth-buddy`. Dated runs land in `projects/`. Repo-specific hints in `.gaslamp/repo_profile.md` and `~/.gaslamp/user.md`.

## Acceptance checklist

1. `python scripts/00_check_env.py` passes
2. Convert one small GGUF → `models/mlx/...`
3. `02_smoke_infer.py` prints coherent text
4. Train 100 steps → `outputs/adapters/`
5. `04_eval_compare.py` shows base vs fine-tuned difference
6. `./scripts/05_export.sh` merges once

## Layout

```
models/gguf/     raw GGUF downloads (gitignored)
models/mlx/      converted checkpoints (gitignored)
data/train.jsonl training data
configs/         hyperparameter YAML
scripts/         pipeline scripts 00–05
projects/        unsloth-buddy dated runs
outputs/         adapters, merged, gguf (gitignored)
```

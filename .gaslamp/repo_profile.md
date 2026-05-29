# local-mlx-tune — repo profile for unsloth-buddy / Gaslamp

## Backend

- **Platform:** Apple Silicon (M1, 8 GB unified RAM)
- **Trainer:** mlx-tune (not Unsloth CUDA)
- **Python:** 3.12 in `.venv/` at repo root

## Conventions

- **Project runs:** `projects/{name}_{date}/` (never repo root)
- **Adapters:** `outputs/adapters/` or `projects/{name}_{date}/outputs/adapters/`
- **Config:** `configs/sft_unambitious.yaml`
- **Training script template:** `scripts/03_train_sft.py`
- **Export script:** `scripts/05_export.sh`

## Model limits (8 GB)

- Default max model size: **1.5B** (4-bit)
- Training: SFT LoRA only; defer GRPO/DPO/vision to Colab
- `max_steps`: 50–200, `batch_size`: 1, `r`: 8, `max_seq_length`: 512–1024

## GGUF ingest

- Raw GGUF → `models/gguf/`
- Convert: `./scripts/01_gguf2mlx.sh <gguf> models/mlx/<name>`
- Re-quantize: `mlx_lm.convert -q` → use `*-4bit` path for training

## Fast path

- Skip GGUF when `mlx-community/*-4bit` on Hugging Face fits the task
- Smoke: `python scripts/02_smoke_infer.py --model <path-or-repo>`

## Export caveat

- GGUF from 4-bit base may fail — merge to f16 first, then `save_pretrained_gguf` or `EXPORT_GGUF=1 ./scripts/05_export.sh`

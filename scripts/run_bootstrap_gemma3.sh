#!/usr/bin/env bash
# Unattended first run: DavidAU safetensors → MLX 4-bit → smoke → short SFT.
# (GGUF/gguf2mlx path is unreliable for Gemma 3 — see configs/models.md)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${LOG:-logs/bootstrap-gemma3.log}"
mkdir -p logs models/mlx outputs

exec > >(tee -a "$LOG") 2>&1

echo "=== bootstrap started $(date) ==="
echo "Log: $LOG"

# shellcheck disable=SC1091
source .venv/bin/activate

HF_MODEL="DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking"
MLX_4BIT="models/mlx/gemma3-1b-heretic-4bit"
TRAIN_STEPS="${TRAIN_STEPS:-100}"

echo "--- Step 0: env check ---"
python scripts/00_check_env.py || true

echo "--- Step 1: mlx_lm convert DavidAU → 4-bit MLX (~2 GB download once) ---"
if [[ -f "$MLX_4BIT/config.json" ]]; then
  echo "Already converted: $MLX_4BIT"
else
  mlx_lm.convert --model "$HF_MODEL" -q --mlx-path "$MLX_4BIT"
fi

echo "--- Step 2: smoke inference ---"
python scripts/02_smoke_infer.py \
  --model "$MLX_4BIT" \
  --prompt "Say hello in one blunt sentence." \
  --max-tokens 48

echo "--- Step 3: unambitious SFT (${TRAIN_STEPS} steps) ---"
python scripts/03_train_sft.py --model "$MLX_4BIT" --max-steps "$TRAIN_STEPS"

echo "--- Step 4: eval base vs adapter ---"
python scripts/04_eval_compare.py --model "$MLX_4BIT"

echo "=== bootstrap finished $(date) ==="
echo "Adapters: outputs/adapters/"
echo "Next: ./scripts/05_export.sh  (optional merge/GGUF)"

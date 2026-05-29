#!/usr/bin/env bash
# Resume bootstrap after model is already converted (skips GGUF path).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${LOG:-logs/bootstrap-resume.log}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
MLX_4BIT="${MLX_4BIT:-models/mlx/gemma3-1b-heretic-4bit}"

mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "=== resume started $(date) ==="
source .venv/bin/activate

if [[ ! -f "$MLX_4BIT/config.json" ]]; then
  echo "Missing $MLX_4BIT — run:"
  echo "  mlx_lm.convert --model DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking -q --mlx-path $MLX_4BIT"
  exit 1
fi

python scripts/02_smoke_infer.py --model "$MLX_4BIT" --prompt "Say hello in one blunt sentence." --max-tokens 48
python scripts/03_train_sft.py --model "$MLX_4BIT" --max-steps "$TRAIN_STEPS"
python scripts/04_eval_compare.py --model "$MLX_4BIT"

echo "=== resume finished $(date) ==="

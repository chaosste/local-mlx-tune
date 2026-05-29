#!/usr/bin/env bash
# 30-minute LoRA run — invoke when Stephen says go.
#
#   cd ~/Projects/local-mlx-tune && source .venv/bin/activate
#   ./scripts/run_train_30min.sh
#
# Monitor: tail -f logs/train-30min.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="logs/train-30min.log"
CONFIG="configs/sft_30min.yaml"
MODEL="models/mlx/gemma3-1b-heretic-4bit"
OUT="outputs/runs/gemma3-30min"

mkdir -p logs outputs/runs

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Missing $MODEL — convert first:"
  echo "  mlx_lm.convert --model DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking -q --mlx-path $MODEL"
  exit 1
fi

ROWS=$(wc -l < data/train.jsonl | tr -d ' ')
if [[ "$ROWS" -lt 20 ]]; then
  echo "WARNING: data/train.jsonl has only ${ROWS} rows."
  echo "  A 30-minute run will overfit. Add 50–200+ examples for a real fine-tune."
  echo "  Proceeding in 5s — Ctrl-C to abort..."
  sleep 5
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== 30-min train started $(date) ==="
echo "Config: $CONFIG"
echo "Output: $OUT"
echo "Log:    $LOG"

source .venv/bin/activate

python scripts/03_train_sft.py --config "$CONFIG"

echo "--- post-train eval (base vs adapter) ---"
python scripts/04_eval_compare.py \
  --model "$MODEL" \
  --adapter-path "$OUT"

echo "=== 30-min train finished $(date) ==="
echo "Adapters: $OUT"
echo "Merge:    MODEL=$MODEL ADAPTER_PATH=$OUT MERGED=outputs/runs/gemma3-30min-merged ./scripts/05_export.sh"

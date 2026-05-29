#!/usr/bin/env bash
# mlx_lm.lora fallback — often more stable on M1 under load than mlx-tune native path.
#
#   cd ~/Projects/local-mlx-tune && source .venv/bin/activate
#   ./scripts/run_train_500_lora.sh
#
# Monitor: tail -f logs/train-500-lora.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="logs/train-500-lora.log"
MODEL="models/mlx/gemma3-1b-heretic-4bit"
OUT="outputs/runs/gemma3-500-2048-lora"
DATA="data"

mkdir -p logs "$OUT"

cp -f data/train.jsonl data/valid.jsonl

exec > >(tee -a "$LOG") 2>&1
echo "=== mlx_lm.lora 500-step run started $(date) ==="
echo "Model:  $MODEL"
echo "Output: $OUT"
echo "Log:    $LOG"

source .venv/bin/activate

mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --adapter-path "$OUT" \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --iters 500 \
  --learning-rate 1.5e-4 \
  --max-seq-length 2048 \
  --save-every 100 \
  --steps-per-report 10 \
  --steps-per-eval 500 \
  --val-batches 0 \
  --mask-prompt \
  --grad-checkpoint \
  --num-layers -1

echo "=== finished $(date) ==="
echo "Chat: mlx_lm.chat --model $MODEL --adapter-path $OUT"

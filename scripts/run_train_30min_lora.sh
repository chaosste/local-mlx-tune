#!/usr/bin/env bash
# 30-minute LoRA via mlx_lm.lora — often survives M1 Metal watchdog better than mlx-tune.
#
# Resume after a crash (once a checkpoint exists):
#   RESUME_ADAPTER=outputs/runs/gemma3-30min-whatis-lora/0000050_adapters.safetensors \
#     ./scripts/run_train_30min_lora.sh
#
#   cd ~/Projects/local-mlx-tune && source .venv/bin/activate
#   ./scripts/run_train_30min_lora.sh
#
# Monitor: tail -f logs/train-30min-lora.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="logs/train-30min-lora.log"
MODEL="models/mlx/gemma3-1b-heretic-4bit"
OUT="outputs/runs/gemma3-30min-whatis-lora"
DATA="data"
ITERS="${ITERS:-1650}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"

mkdir -p logs "$OUT"
cp -f data/train.jsonl data/valid.jsonl

exec > >(tee -a "$LOG") 2>&1
echo "=== mlx_lm.lora 30-min run started $(date) ==="
echo "Model:  $MODEL"
echo "Output: $OUT"
echo "Iters:  $ITERS"
echo "Log:    $LOG"
if [[ -n "$RESUME_ADAPTER" ]]; then
  echo "Resume: $RESUME_ADAPTER"
fi

source .venv/bin/activate

LORA_CMD=(
  mlx_lm.lora
  --model "$MODEL"
  --train
  --data "$DATA"
  --adapter-path "$OUT"
  --batch-size 1
  --grad-accumulation-steps 4
  --iters "$ITERS"
  --learning-rate 1.5e-4
  --max-seq-length 1024
  --save-every 50
  --steps-per-report 10
  --steps-per-eval "$ITERS"
  --val-batches 0
  --mask-prompt
  --grad-checkpoint
  --num-layers -1
)

if [[ -n "$RESUME_ADAPTER" ]]; then
  LORA_CMD+=(--resume-adapter-file "$RESUME_ADAPTER")
fi

if [[ "${CAFFEINATE:-1}" == "1" ]]; then
  echo "caffeinate: idle sleep blocked for training"
  caffeinate -i "${LORA_CMD[@]}"
else
  "${LORA_CMD[@]}"
fi

echo "=== finished $(date) ==="
echo "Chat: mlx_lm.chat --model $MODEL --adapter-path $OUT"

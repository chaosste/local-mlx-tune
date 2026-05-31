#!/usr/bin/env bash
# ~500-step LoRA run — max_seq_length 2048, 53-row train.jsonl
#
#   cd ~/Projects/local-mlx-tune && source .venv/bin/activate
#   ./scripts/run_train_500.sh
#   # or overnight wrapper (caffeinate on by default):
#   ./scripts/run_train_overnight.sh
#
# Monitor: tail -f logs/train-500.log
# Opt out of caffeinate: CAFFEINATE=0 ./scripts/run_train_500.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="logs/train-500.log"
CONFIG="configs/sft_500steps.yaml"
MODEL="models/mlx/gemma3-1b-heretic-4bit"
OUT="outputs/runs/gemma3-discourse"

mkdir -p logs outputs/runs

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Missing $MODEL — convert first:"
  echo "  mlx_lm.convert --model DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking -q --mlx-path $MODEL"
  exit 1
fi

ROWS=$(wc -l < data/train.jsonl | tr -d ' ')
echo "Training rows: ${ROWS}"

exec > >(tee -a "$LOG") 2>&1
echo "=== 500-step train started $(date) ==="
echo "Config: $CONFIG"
echo "Output: $OUT"
echo "Log:    $LOG"

source .venv/bin/activate

if [[ "${CAFFEINATE:-1}" == "1" ]]; then
  echo "caffeinate: idle sleep blocked for training"
  caffeinate -i python scripts/03_train_sft.py --config "$CONFIG"
else
  python scripts/03_train_sft.py --config "$CONFIG"
fi

echo "--- post-train eval (base vs adapter) ---"
python scripts/04_eval_compare.py \
  --model "$MODEL" \
  --adapter-path "$OUT"

echo "=== 500-step train finished $(date) ==="
echo "Adapters: $OUT"
echo "Merge:    MODEL=$MODEL ADAPTER_PATH=$OUT MERGED=outputs/runs/gemma3-discourse-merged ./scripts/05_export.sh"

#!/usr/bin/env bash
# Glossary-only LoRA — short run on data/glossary.jsonl → outputs/runs/gemma3-glossary
#
#   cd ~/Projects/local-mlx-tune && source .venv/bin/activate
#   ./scripts/run_train_glossary.sh
#
# Resume: RESUME_ADAPTER=outputs/runs/gemma3-glossary/0000050_adapters.safetensors ./scripts/run_train_glossary.sh
# Monitor: tail -f logs/train-glossary.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="logs/train-glossary.log"
CONFIG="configs/sft_glossary.yaml"
MODEL="models/mlx/gemma3-1b-heretic-4bit"
OUT="outputs/runs/gemma3-glossary"
DATA="data"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"

mkdir -p logs "$OUT"
cp -f data/glossary.jsonl data/valid.jsonl

exec > >(tee -a "$LOG") 2>&1
echo "=== glossary train started $(date) ==="
echo "Config: $CONFIG"
echo "Output: $OUT"
echo "Rows:   $(wc -l < data/glossary.jsonl | tr -d ' ')"
echo "Log:    $LOG"

source .venv/bin/activate

STEPS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['max_steps'])")
SEQ=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['max_seq_length'])")

LORA_CMD=(
  mlx_lm.lora
  --model "$MODEL"
  --train
  --data "$DATA"
  --adapter-path "$OUT"
  --batch-size 1
  --grad-accumulation-steps 4
  --iters "$STEPS"
  --learning-rate 1.5e-4
  --max-seq-length "$SEQ"
  --save-every 50
  --steps-per-report 10
  --steps-per-eval "$STEPS"
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

echo "=== glossary train finished $(date) ==="
echo "Chat:  mlx_lm.chat --model $MODEL --adapter-path $OUT"
echo "Eval:  python scripts/04_eval_dual_adapters.py"

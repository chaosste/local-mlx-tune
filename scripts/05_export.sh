#!/usr/bin/env bash
# Merge LoRA adapters and optionally export GGUF for Ollama/llama.cpp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/mlx/my-uncensored-base-4bit}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/adapters}"
MERGED="${MERGED:-outputs/merged}"
EXPORT_GGUF="${EXPORT_GGUF:-0}"

echo "=== Merge adapters ==="
echo "Base:     $MODEL"
echo "Adapters: $ADAPTER_PATH"
echo "Output:   $MERGED"
echo ""

python3 - <<PY
from pathlib import Path
from mlx_tune import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "${MODEL}",
    max_seq_length=1024,
    load_in_4bit=True,
)
model.load_adapter("${ADAPTER_PATH}")
out = Path("${MERGED}")
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained_merged(str(out), tokenizer)
print(f"Merged model saved to {out}")
PY

if [[ "$EXPORT_GGUF" == "1" ]]; then
  echo ""
  echo "=== GGUF export ==="
  echo "Note: GGUF export from 4-bit bases may fail. Merge to f16 first if needed."
  python3 - <<PY
from pathlib import Path
from mlx_tune import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("${MERGED}", max_seq_length=1024)
gguf_dir = Path("outputs/gguf")
gguf_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained_gguf(str(gguf_dir), tokenizer)
print(f"GGUF written under {gguf_dir}")
PY
else
  echo ""
  echo "Skip GGUF export (set EXPORT_GGUF=1 to enable)."
  echo "Alternative: mlx_lm.fuse --model $MODEL --adapter-path $ADAPTER_PATH --export-gguf"
fi

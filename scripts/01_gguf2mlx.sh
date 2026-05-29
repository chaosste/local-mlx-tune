#!/usr/bin/env bash
# Convert a GGUF checkpoint to HuggingFace-layout safetensors for mlx_lm.load().
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-}"
OUTPUT="${2:-models/mlx/my-uncensored-base}"
DTYPE="${DTYPE:-float16}"

if [[ -z "$INPUT" ]]; then
  echo "Usage: $0 <path-to.gguf> [output-dir]"
  echo ""
  echo "Examples:"
  echo "  $0 models/gguf/model-Q4_K_M.gguf models/mlx/my-uncensored-base"
  echo "  DTYPE=float32 $0 models/gguf/model.gguf models/mlx/my-base-f32"
  echo ""
  echo "Inspect metadata only:"
  echo "  gguf2mlx -i models/gguf/model.gguf --skip-weights"
  exit 1
fi

if ! command -v gguf2mlx >/dev/null 2>&1; then
  echo "gguf2mlx not found. Install with:"
  echo "  source .venv/bin/activate"
  echo "  uv pip install 'gguf2mlx @ git+https://github.com/barrontang/gguf2mlx.git'"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "Converting $INPUT → $OUTPUT (dtype=$DTYPE)"
gguf2mlx --input "$INPUT" --output "$OUTPUT" --dtype "$DTYPE"

echo ""
echo "Optional: re-quantize for QLoRA training (saves RAM on 8GB Macs):"
echo "  mlx_lm.convert --model $OUTPUT -q --mlx-path ${OUTPUT}-4bit"

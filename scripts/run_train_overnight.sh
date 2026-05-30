#!/usr/bin/env bash
# Overnight / unattended LoRA — wraps run_train_500.sh with caffeinate.
#
# Monitor-off is fine; close heavy GPU apps (browser video, etc.) before starting.
# Disable caffeinate: CAFFEINATE=0 ./scripts/run_train_overnight.sh
#
#   cd ~/Projects/local-mlx-tune && source .venv/bin/activate
#   ./scripts/run_train_overnight.sh
#
# Monitor: tail -f logs/train-500.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CAFFEINATE="${CAFFEINATE:-1}"
exec "$ROOT/scripts/run_train_500.sh"

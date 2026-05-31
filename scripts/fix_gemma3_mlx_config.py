#!/usr/bin/env python3
"""Patch gguf2mlx Gemma 3 output so mlx-lm can load and quantize it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fix_config(config: dict) -> dict:
    out = dict(config)
    out["model_type"] = "gemma3_text"
    out.setdefault("head_dim", 256)
    out.setdefault("query_pre_attn_scalar", 256)
    out.setdefault("sliding_window", 512)
    out.setdefault("sliding_window_pattern", 6)
    out.setdefault("rope_local_base_freq", 10000.0)
    out.setdefault("hidden_activation", "gelu_pytorch_tanh")
    out.setdefault("rope_scaling", None)
    out.setdefault("pad_token_id", 0)
    if out.get("rms_norm_eps") == 9.999999974752427e-07:
        out["rms_norm_eps"] = 1e-6
    if out.get("attention_bias") is True:
        out["attention_bias"] = False
    for key in list(out):
        if key.startswith("_gguf"):
            out.pop(key, None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix Gemma 3 MLX config after gguf2mlx")
    parser.add_argument(
        "model_dir",
        type=Path,
        nargs="?",
        default=Path("models/mlx/gemma3-1b-heretic"),
    )
    args = parser.parse_args()
    config_path = args.model_dir / "config.json"
    if not config_path.exists():
        print(f"Missing {config_path}", file=sys.stderr)
        return 1
    config = json.loads(config_path.read_text())
    fixed = fix_config(config)
    config_path.write_text(json.dumps(fixed, indent=2) + "\n")
    print(f"Patched {config_path} (model_type=gemma3_text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Smoke-test inference on a converted or Hub MLX model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="MLX smoke inference")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models/mlx/my-uncensored-base"),
        help="Local path or HuggingFace repo id",
    )
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    from mlx_lm import generate, load

    print(f"Loading: {args.model}")
    model, tokenizer = load(args.model)
    print(f"Prompt: {args.prompt!r}\n")
    text = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare base model vs fine-tuned adapter on fixed prompts."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPTS = [
    "Write a one-line greeting.",
    "Explain what MLX is in one sentence.",
    "Give a blunt yes/no: is the sky blue on a clear day?",
]


def generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate as mlx_generate

    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Base vs adapter comparison")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models/mlx/my-uncensored-base-4bit"),
        help="Base model path or HF repo",
    )
    parser.add_argument(
        "--adapter-path",
        default=str(ROOT / "outputs/adapters"),
        help="LoRA adapter directory",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", action="append", dest="prompts", help="Custom prompt (repeatable)")
    args = parser.parse_args()

    from mlx_lm import load

    prompts = args.prompts or DEFAULT_PROMPTS
    adapter_path = Path(args.adapter_path)

    print(f"Base model: {args.model}\n")
    if adapter_path.exists():
        print(f"Adapter:   {adapter_path}\n")
    else:
        print(f"Adapter:   {adapter_path} (missing — showing base only)\n")

    base_model, tokenizer = load(args.model)
    for i, prompt in enumerate(prompts, 1):
        print(f"--- Prompt {i} ---")
        print(f"Q: {prompt}\n")
        print(f"[Base]       {generate(base_model, tokenizer, prompt, args.max_tokens)}")
        print()
    del base_model
    gc.collect()

    if adapter_path.exists():
        tuned_model, tokenizer = load(args.model, adapter_path=str(adapter_path))
        for i, prompt in enumerate(prompts, 1):
            print(f"--- Prompt {i} (Fine-tuned) ---")
            print(f"Q: {prompt}\n")
            print(f"[Fine-tuned] {generate(tuned_model, tokenizer, prompt, args.max_tokens)}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

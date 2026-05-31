#!/usr/bin/env python3
"""Compare two LoRA adapters on the same prompts (chat-formatted)."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPTS = [
    "What is 'the shadow'?",
    "What is 'cognitive liberty'?",
]

DEFAULT_ADAPTERS = {
    "discourse": ROOT / "outputs/runs/gemma3-discourse",
    "glossary": ROOT / "outputs/runs/gemma3-glossary",
}


def chat_prompt(tokenizer, user_text: str, system: str | None) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_chat(model, tokenizer, user_text: str, max_tokens: int, system: str | None) -> str:
    from mlx_lm import generate as mlx_generate

    prompt = chat_prompt(tokenizer, user_text, system)
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens).strip()


def load_adapter_model(model_path: str, adapter_path: Path):
    from mlx_lm import load

    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")
    return load(model_path, adapter_path=str(adapter_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Side-by-side adapter comparison (chat template)")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models/mlx/gemma3-1b-heretic-4bit"),
        help="Base model path or HF repo",
    )
    parser.add_argument(
        "--adapter-a",
        type=Path,
        default=DEFAULT_ADAPTERS["discourse"],
        help="First adapter directory",
    )
    parser.add_argument(
        "--label-a",
        default="discourse",
        help="Display label for adapter A",
    )
    parser.add_argument(
        "--adapter-b",
        type=Path,
        default=DEFAULT_ADAPTERS["glossary"],
        help="Second adapter directory",
    )
    parser.add_argument(
        "--label-b",
        default="glossary",
        help="Display label for adapter B",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--system",
        help='Optional system prompt (e.g. "Answer in 3–4 sentences. Define the term only.")',
    )
    parser.add_argument("--prompt", action="append", dest="prompts", help="User prompt (repeatable)")
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    adapters = [
        (args.label_a, args.adapter_a),
        (args.label_b, args.adapter_b),
    ]

    print(f"Base model: {args.model}")
    for label, path in adapters:
        print(f"  [{label}] {path}")
    if args.system:
        print(f"System: {args.system!r}")
    print()

    responses: dict[str, dict[str, str]] = {label: {} for label, _ in adapters}

    for label, adapter_path in adapters:
        print(f"Loading [{label}]...", flush=True)
        model, tokenizer = load_adapter_model(args.model, adapter_path)
        for prompt in prompts:
            responses[label][prompt] = generate_chat(
                model, tokenizer, prompt, args.max_tokens, args.system
            )
        del model
        gc.collect()

    for i, prompt in enumerate(prompts, 1):
        print(f"--- Prompt {i} ---")
        print(f"Q: {prompt}\n")
        for label, _ in adapters:
            text = responses[label][prompt]
            print(f"[{label}]")
            print(text)
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

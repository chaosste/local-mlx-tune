#!/usr/bin/env python3
"""Unambitious SFT LoRA via mlx-tune (Unsloth-compatible API)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="mlx-tune SFT (unambitious defaults)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sft_unambitious.yaml",
    )
    parser.add_argument("--model", help="Override model path or HF repo")
    parser.add_argument("--max-steps", type=int, help="Override max_steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or cfg["model"]
    max_steps = args.max_steps or cfg["training"]["max_steps"]
    lora = cfg["lora"]
    train_cfg = cfg["training"]

    from datasets import load_dataset
    from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer

    print(f"Model: {model_path}")
    print(f"Data:  {cfg['data']['train_file']}")
    print(f"Steps: {max_steps}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", True),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        target_modules=lora["target_modules"],
    )

    train_file = ROOT / cfg["data"]["train_file"]
    dataset = load_dataset("json", data_files=str(train_file), split="train")

    output_dir = ROOT / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            max_steps=max_steps,
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
        ),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    print(f"Adapters saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

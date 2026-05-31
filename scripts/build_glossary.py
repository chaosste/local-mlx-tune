#!/usr/bin/env python3
"""Build glossary JSONL + review markdown from WHAT_IS source (defs only, no examples)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from merge_what_is import parse_what_is_md, write_jsonl  # noqa: E402


def strip_example(answer: str) -> str:
    answer = re.sub(r"\s*\n\nExample:.*", "", answer, flags=re.S)
    answer = re.sub(r"\.\s*Example:.*", ".", answer, flags=re.S)
    return answer.strip()


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "# Glossary training set (review)",
        "",
        "Definitions only — `Example:` blocks removed for glossary LoRA.",
        "Edit this file or `data/raw/WHAT_IS.md`, then re-run `scripts/build_glossary.py`.",
        "",
    ]
    for i, row in enumerate(rows, 1):
        q = row["messages"][0]["content"]
        a = row["messages"][1]["content"]
        lines.append(f"## {i}. {q}")
        lines.append("")
        lines.append(a)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/glossary.jsonl from WHAT_IS source")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/raw/WHAT_IS.md",
    )
    parser.add_argument("--jsonl", type=Path, default=ROOT / "data/glossary.jsonl")
    parser.add_argument("--markdown", type=Path, default=ROOT / "data/glossary.md")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Missing source: {args.source}")

    rows = parse_what_is_md(args.source)
    glossary: list[dict] = []
    for row in rows:
        q = row["messages"][0]["content"]
        a = strip_example(row["messages"][1]["content"])
        if not a:
            continue
        glossary.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    write_jsonl(args.jsonl, glossary)
    args.markdown.write_text(to_markdown(glossary))
    print(f"wrote {len(glossary)} rows -> {args.jsonl}")
    print(f"wrote review copy -> {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

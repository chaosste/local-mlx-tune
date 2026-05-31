#!/usr/bin/env python3
"""Merge WHAT_IS.md rows into data/train.jsonl as chat JSONL."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize_question(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^q:\s*", "", t)
    t = re.sub(r"^a:\s*", "", t)
    t = (
        t.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    t = re.sub(r"['\"`]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.rstrip("?").strip()


def clean_question(text: str) -> str:
    q = text.strip()
    q = re.sub(r"^\*\*", "", q)
    q = re.sub(r"\*\*$", "", q)
    q = re.sub(r"^Q:\s*", "", q, flags=re.I)
    return q.strip()


def clean_answer(text: str) -> str:
    a = text.strip()
    a = re.sub(r"^A:\s*", "", a, flags=re.I)
    a = a.replace("<br><br>", "\n\n").replace("<br>", "\n")
    return a.strip()


def parse_what_is_md(path: Path) -> list[dict]:
    text = path.read_text()
    rows: list[tuple[str, str]] = []

    for line in text.splitlines():
        if not line.strip().startswith("|") or line.count("|") < 3:
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) != 2:
            continue
        q, a = parts
        if not q or not a or q.lower() == "concept" or set(q) <= {"-"}:
            continue
        if normalize_question(q) == normalize_question(
            'Why do scholars warn that using "decolonization" as a mere metaphor for the psychedelic experience is highly problematic?'
        ):
            # Source row has a duplicated/wrong answer; skip until corrected.
            continue
        rows.append((clean_question(q), clean_answer(a)))

    # Row split across lines in source table.
    colonial_ego = re.search(
        r"\|\s*How exactly do psychedelics disrupt the \"colonial ego\"\?\s*\|\s*(.+?)\s*\|",
        text,
        flags=re.S,
    )
    if colonial_ego:
        rows.append(
            (
                'How exactly do psychedelics disrupt the "colonial ego"?',
                clean_answer(colonial_ego.group(1)),
            )
        )

    for block in re.findall(
        r"\*\*Q:\s*(.+?)\*\*\s*\n\*\*A:\*\*\s*(.+?)(?=\n\*\*Q:|\n\*\*What are|\Z)",
        text,
        flags=re.S,
    ):
        rows.append((clean_question(block[0]), clean_answer(block[1])))

    for block in re.findall(
        r"\*\*(What are the risks of.+?)\*\*\s*\n(.+?)(?=\n\*\*Q:|\Z)",
        text,
        flags=re.S,
    ):
        rows.append((clean_question(block[0]), clean_answer(block[1])))

    out: list[dict] = []
    seen: set[str] = set()
    for q, a in rows:
        key = normalize_question(q)
        if key in seen or not a:
            continue
        seen.add(key)
        out.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_train(existing: list[dict], what_is: list[dict]) -> tuple[list[dict], dict]:
    by_q: dict[str, dict] = {}
    order: list[str] = []

    for row in existing:
        q = row["messages"][0]["content"]
        key = normalize_question(q)
        by_q[key] = row
        order.append(key)

    replaced = 0
    added = 0
    for row in what_is:
        q = row["messages"][0]["content"]
        key = normalize_question(q)
        if key in by_q:
            by_q[key] = row
            replaced += 1
        else:
            by_q[key] = row
            order.append(key)
            added += 1

    merged = [by_q[k] for k in order]
    stats = {"replaced": replaced, "added": added, "total": len(merged)}
    return merged, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge WHAT_IS.md into train.jsonl")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Users/stephenbeale/Desktop/WHAT_IS/WHAT_IS.md"),
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.jsonl")
    parser.add_argument("--what-is-out", type=Path, default=ROOT / "data/what_is.jsonl")
    parser.add_argument("--raw-copy", type=Path, default=ROOT / "data/raw/WHAT_IS.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Missing source: {args.source}")

    what_is_rows = parse_what_is_md(args.source)
    existing = load_jsonl(args.train)

    merged, stats = merge_train(existing, what_is_rows)

    print(f"WHAT_IS parsed: {len(what_is_rows)} rows")
    print(f"train.jsonl before: {len(existing)} rows")
    print(f"replaced: {stats['replaced']}, added: {stats['added']}, total: {stats['total']}")

    if args.dry_run:
        print("\nNew WHAT_IS questions:")
        for row in what_is_rows:
            print(" -", row["messages"][0]["content"])
        return 0

    args.raw_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.raw_copy)
    write_jsonl(args.what_is_out, what_is_rows)
    backup = args.train.with_suffix(".jsonl.bak")
    shutil.copy2(args.train, backup)
    write_jsonl(args.train, merged)
    print(f"wrote: {args.what_is_out}")
    print(f"wrote: {args.train} (backup: {backup})")
    print(f"copied source -> {args.raw_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

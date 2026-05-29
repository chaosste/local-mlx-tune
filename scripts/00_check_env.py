#!/usr/bin/env python3
"""Verify Apple Silicon MLX environment before training."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check_arm_python() -> bool:
    proc = platform.processor()
    ok = proc == "arm"
    print(f"Python processor: {proc!r} {'OK' if ok else 'FAIL (need native arm64)'}")
    return ok


def check_mlx() -> bool:
    try:
        import mlx.core as mx

        device = mx.default_device()
        print(f"MLX default device: {device} OK")
        return True
    except ImportError as exc:
        print(f"MLX import failed: {exc}")
        return False


def check_packages() -> bool:
    required = ["mlx_lm", "mlx_tune", "datasets", "yaml"]
    ok = True
    for name in required:
        try:
            __import__(name)
            print(f"Package {name}: OK")
        except ImportError:
            print(f"Package {name}: MISSING")
            ok = False
    return ok


def check_gguf2mlx() -> bool:
    if shutil.which("gguf2mlx"):
        print("gguf2mlx CLI: OK")
        return True
    print("gguf2mlx CLI: not on PATH (install with: uv pip install -e '.[gguf]')")
    return False


def check_disk(min_gb: float = 30.0) -> bool:
    usage = shutil.disk_usage(Path(__file__).resolve().parents[1])
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    print(f"Free disk: {free_gb:.1f} GB {'OK' if ok else f'WARN (< {min_gb} GB for 1B conversion)'}")
    return ok


def check_ram() -> None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        gb = int(out) / (1024**3)
        print(f"Unified RAM: {gb:.0f} GB")
        if gb <= 8:
            print("  → Use ≤1.5B models, 4-bit base, batch-size 1, max_steps ~100")
    except (subprocess.CalledProcessError, ValueError):
        print("Unified RAM: unknown")


def main() -> int:
    print("=== local-mlx-tune environment check ===\n")
    checks = [
        check_arm_python(),
        check_mlx(),
        check_packages(),
        check_gguf2mlx(),
        check_disk(),
    ]
    check_ram()
    print()
    if all(checks[:3]):
        print("Core stack ready.")
        return 0
    print("Fix missing items above, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

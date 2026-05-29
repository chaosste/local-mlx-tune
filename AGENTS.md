## Learned User Preferences

- Prefers concise, action-oriented responses without excessive caveats or repeated warnings.
- Start long training runs only on explicit user signal (e.g. "on my mark"), not proactively.
- Wants a real calibration dataset in place before meaningful SFT runs, not placeholder JSONL.
- Stop immediately when asked; do not retry hung tool calls without checking in first.
- Prefer lightweight Hugging Face dataset discovery (CLI or single call) over multi-call MCP searches that timeout.

## Learned Workspace Facts

- Apple M1 Mac with 8 GB unified RAM; Python 3.12 venv at `~/Projects/local-mlx-tune/.venv`.
- Apple Silicon MLX LoRA pipeline using mlx-lm + mlx-tune, optionally orchestrated via unsloth-buddy.
- Default training base: Gemma 3 1B Heretic 4-bit at `models/mlx/gemma3-1b-heretic-4bit`.
- For Gemma 3, convert `DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking` via `mlx_lm.convert`; gguf2mlx is broken for Gemma 3 (config/weight mismatch).
- Treat ≤1.5B 4-bit models as default on 8 GB; SFT LoRA locally, larger models or GRPO/DPO on Colab + Unsloth.
- Project goal: uncensored Hub bases plus small SFT to steer style/task behavior, not jailbreak-from-scratch training.
- Primary configs: `configs/sft_unambitious.yaml` (100-step default), `configs/sft_30min.yaml` (~1650 steps).
- Bootstrap/resume scripts: `run_bootstrap_gemma3.sh`, `run_bootstrap_resume.sh`, `run_train_30min.sh`.

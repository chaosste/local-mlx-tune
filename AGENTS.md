## Learned User Preferences

- Prefers concise, action-oriented responses without excessive caveats or repeated warnings.
- Start long training runs only on explicit user signal (e.g. "on my mark"), not proactively.
- Wants a real calibration dataset in place before meaningful SFT runs, not placeholder JSONL.
- Stop immediately when asked; do not retry hung tool calls without checking in first.
- Prefer lightweight Hugging Face dataset discovery (CLI or single call) over multi-call MCP searches that timeout.
- Do not suggest GPU, Azure GPU, CUDA, or cloud GPU as default fallbacks; access unavailable (quota/signup blocks). Paid Colab Pro only if user explicitly opts in.
- Defer conventional HF/Unsloth personalisation on aligned models until the local MLX uncensored pipeline is validated.
- Provide shell commands without inline `#` comments; zsh interactive sessions treat `#` as arguments unless `setopt interactivecomments`.
- Do not invent architecture, machine roles, parallel workflows, or optional layers; keep responses and repo guidance minimal unless the user asks.

## Learned Workspace Facts

- Apple M1 Mac (8 GB unified RAM); Python 3.12 venv at `~/Projects/local-mlx-tune/.venv`; activate before any `python` / `mlx_lm.*`. MLX LoRA via mlx-lm + mlx-tune (unsloth-buddy optional).
- Local MLX LoRA on M1 8 GB is the only training path; Azure ML `third-vm` is Standard_E4ds_v4 (CPU-only, no GPU). Treat ≤1.5B 4-bit models as default.
- Default training base: Gemma 3 1B Heretic 4-bit at `models/mlx/gemma3-1b-heretic-4bit` — this IS MLX LoRA on a quantized base; no separate "switch to QLoRA" step.
- For Gemma 3, convert `DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking` via `mlx_lm.convert`; gguf2mlx is broken for Gemma 3 (config/weight mismatch).
- Project goal: uncensored Hub bases plus small SFT to steer style/task behavior, not jailbreak-from-scratch training.
- Train glossary and discourse as separate LoRAs (never one combined run); pick one adapter at chat time.
- Data: `data/train.jsonl` (discourse corpus); `data/glossary.jsonl` (defs-only via `scripts/build_glossary.py`, source `data/raw/WHAT_IS.md`).
- Adapters (see `configs/adapters.md`): `outputs/runs/gemma3-discourse` (`sft_500steps.yaml`, voice); `outputs/runs/gemma3-glossary` (`sft_glossary.yaml`, ~150 steps).
- Bootstrap/run scripts: `run_bootstrap_gemma3.sh`, `run_train_30min.sh`, `run_train_30min_lora.sh`, `run_train_500.sh`, `run_train_glossary.sh` (log: `logs/train-glossary.log`), `run_train_overnight.sh` (caffeinate on by default; lora scripts save every 50 with resume).
- macOS Metal `ImpactingInteractivity` aborts training under interactive GPU load; run idle from Terminal with `caffeinate -i`.
- Git for code; `models/` and `outputs/` are gitignored — rsync them between checkouts when a copy is missing.
- Domain eval: trust `mlx_lm.chat` with adapter; use `04_eval_dual_adapters.py` for discourse vs glossary; not `04_eval_compare.py` (raw completion).

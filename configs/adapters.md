# Adapter naming

Use **purpose names** in paths. Hyperparameters live in configs, not folder names.

| Path | Role | Config | Notes |
|------|------|--------|-------|
| `outputs/runs/gemma3-discourse` | Long-form voice (femtheogenic + decolonising) | `configs/sft_500steps.yaml` | 500 steps, seq 2048 |
| `outputs/runs/gemma3-glossary` | Short "What is X?" definitions | `configs/sft_glossary.yaml` | ~150 steps, seq 1024 |
| `outputs/runs/gemma3-30min-whatis-lora` | **Deprecated experiment** — merged corpus, partial run | — | Prefer glossary + discourse split |

## Rename existing discourse adapter (one-time)

If you trained before this naming scheme:

```bash
mv outputs/runs/gemma3-500-2048 outputs/runs/gemma3-discourse
```

Chat:

```bash
mlx_lm.chat --model models/mlx/gemma3-1b-heretic-4bit \
  --adapter-path outputs/runs/gemma3-discourse
```

Compare (after `source .venv/bin/activate`):

```bash
python scripts/04_eval_dual_adapters.py
```

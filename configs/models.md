# Starting models (M1 8 GB)

Ranked for **uncensored bases** + **unambitious LoRA** in this repo.

| Priority | Name | Hugging Face | Format | 8 GB train? | Pipeline |
|----------|------|--------------|--------|-------------|----------|
| **1** | Gemma 3 1B Heretic Thinking | [mradermacher/...-GGUF](https://huggingface.co/mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-GGUF) | GGUF → MLX | **Yes** | `./scripts/run_bootstrap_gemma3.sh` |
| **2** | Gemma 4 E2B uncensored | [deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit](https://huggingface.co/deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit) | MLX 4-bit | Borderline | smoke + short LoRA only |
| **3** | Gemma 4 E2B uncensored (GGUF) | [TrevorJS/gemma-4-E2B-it-uncensored-GGUF](https://huggingface.co/TrevorJS/gemma-4-E2B-it-uncensored-GGUF) | GGUF → MLX | Borderline | `01_gguf2mlx.sh` then `-q` |
| **4** | Qwen3.5 4B OptiQ | [mlx-community/Qwen3.5-4B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-OptiQ-4bit) | MLX 4-bit | Unlikely OOM | inference / baseline only |
| — | Google Gemma 4 E2B (censored) | [google/gemma-4-e2b](https://huggingface.co/google/gemma-4-e2b) → [mlx-community/gemma-4-e2b-it-4bit](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit) | MLX | Borderline | skip for jailbreak goal |

Default in [`sft_unambitious.yaml`](sft_unambitious.yaml): `models/mlx/gemma3-1b-heretic-4bit`

---

## 1. Gemma 3 1B Heretic Thinking (recommended)

**Sources**

- GGUF: `mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-GGUF` (Q4_K_M or Q5_K_M)
- Safetensors: `DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking`

**Unattended bootstrap (recommended path)**

Uses DavidAU safetensors → `mlx_lm.convert -q` (avoids gguf2mlx Gemma 3 config/weight issues):

```bash
source .venv/bin/activate
./scripts/run_bootstrap_gemma3.sh
tail -f logs/bootstrap-gemma3.log
```

If convert already finished, resume training only:

```bash
chmod +x scripts/run_bootstrap_resume.sh
./scripts/run_bootstrap_resume.sh
```

**GGUF path (optional, currently broken for Gemma 3 in mlx-lm)**

gguf2mlx produces a flat `gemma3` config and orphan `blk.*` weight keys. Use DavidAU safetensors instead until fixed upstream.

```bash
# Manual one-liner (same as bootstrap step 1):
mlx_lm.convert --model DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking \
  -q --mlx-path models/mlx/gemma3-1b-heretic-4bit
```

**Legacy GGUF steps (not recommended for Gemma 3)**

```bash
./scripts/01_gguf2mlx.sh models/gguf/<file>.gguf models/mlx/gemma3-1b-heretic
python scripts/fix_gemma3_mlx_config.py models/mlx/gemma3-1b-heretic  # config only; weights still broken
```

---

## 2. Gemma 4 E2B Heretic uncensored (MLX fast path)

Already MLX — no GGUF step.

```bash
python scripts/02_smoke_infer.py \
  --model deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit

python scripts/03_train_sft.py \
  --model deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit \
  --max-steps 50
```

Multimodal (VLM). Heavier than Gemma 3 1B on 8 GB.

---

## 3. Qwen3.5 4B OptiQ (aligned, not uncensored)

```bash
python scripts/02_smoke_infer.py --model mlx-community/Qwen3.5-4B-OptiQ-4bit
```

Use as quality baseline, not uncensored training base.

---

## 4. google/gemma-4-e2b (censored official base)

```bash
python scripts/02_smoke_infer.py --model mlx-community/gemma-4-e2b-it-4bit
```

Only if you want the stock Google model before any abliteration work.

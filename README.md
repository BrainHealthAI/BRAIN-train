# MedQA — Llama-3.1-8B Fine-Tuning on RunPod

End-to-end pipeline for training **Llama-3.1-8B-Instruct** on
**`Williamsanderson/MedQAData-v2`** with two methods:

| Method | What it does |
|---|---|
| **SFT** (QLoRA) | Classic supervised fine-tuning on chat-formatted triplets (EN/FR/Darija). 1 epoch. |
| **SASR** | Step-wise Adaptive SFT + GRPO (arXiv:2505.13026). Starts from the SFT checkpoint; adaptively alternates between SFT and GRPO rollouts at the optimizer-step level, guided by gradient-norm ratio. |

The pipeline produces a **3-way benchmark** comparison:
*baseline (no fine-tuning) ↔ SFT ↔ SASR* on BERTScore, ROUGE-L, METEOR, BLEU, exact-match — per language and macro-averaged.

---

## 0. Choice of RunPod GPU

### A40 vs L40S for this workload

| GPU | VRAM | BF16 TFLOPs | Memory BW | RunPod price |
|---|---:|---:|---:|---|
| **A40** | 48 GB | 149 | 696 GB/s | ~$0.39/h comm. |
| **L40S** | 48 GB | **362** | **864 GB/s** | ~$0.86/h comm. |
| A100 40GB | 40 GB | 312 | 1 555 GB/s | ~$1.89/h comm. |

**Recommendation: L40S.** For QLoRA fine-tuning of an 8B model with short-ish sequences (2048 max),
we are **compute-bound, not memory-bound**. L40S has **~2.4× the BF16 throughput of A40**, so for
roughly 2× the hourly cost you get the job done in ~45% of the A40 time → **L40S is cheaper in total $**.

If you want to minimize $ and don't mind waiting, **A40** is fine (same 48 GB VRAM; the 4090's 24 GB is tight).
If you want the fastest training with the most headroom, **L40S**.

**A100 40 GB** is not worth it vs L40S for this workload (lower TFLOPs, higher price, less VRAM).
Pick A100 80 GB only if you want to train without QLoRA (full bf16 LoRA).

### Recommended pod image

```
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

Container disk ≥ 30 GB, Volume ≥ 50 GB (Llama-3.1-8B weights are ~16 GB).

---

## 1. One-time setup on the pod

```bash
# Open the RunPod web terminal, then:
cd /workspace
git clone <this-repo>   # OR upload the runpod_training/ folder via the file browser
mv runpod_training scripts
cd scripts
chmod +x setup_runpod.sh launch.sh

bash setup_runpod.sh
```

Then set your tokens (get them from HF + wandb UIs):

```bash
export HF_TOKEN="hf_xxxxx"                # from https://huggingface.co/settings/tokens
export WANDB_API_KEY="wandb_xxxxx"        # from https://wandb.ai/authorize

huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_API_KEY
```

> ⚠ **Llama-3.1 is gated.** You must have accepted the license on the HuggingFace model page with the same account the token comes from, otherwise the model download will fail.

---

## 2. Run the full pipeline

```bash
cd /workspace/scripts
bash launch.sh                            # full 35K rows (real run)
# -- or for a smoke test first --
LIMIT=500 EVAL_SAMPLES=30 bash launch.sh  # ~30 min total, validates everything
```

### What happens

1. **`evaluate.py` (baseline)** — benchmarks the untouched `Llama-3.1-8B-Instruct` on a held-out eval set (100 samples × 3 languages by default).
2. **`train_sft.py`** — QLoRA SFT, 1 epoch, wandb-tracked. Saves adapters to `/workspace/models/medqa-sft` + records `‖∇L_SFT‖` into `sft_meta.json` (used as `G_warmup` fallback).
3. **`train_sasr.py`** — Loads the SFT adapter, runs a warm-up phase (200 pure SFT steps) to record the true `G_warmup`, then switches to the adaptive phase. At each optimizer step:
   - Compute `p_t = G_last / (G_last + γ·G_warmup)`  (**Eq. 12 from the paper**)
   - Sample α ~ U(0,1); if α < p → SFT step (update `G_last`), else → GRPO step (G rollouts, ROUGE-L reward, group-relative advantages, clipped surrogate + KL-to-SFT penalty).
4. **`evaluate.py` (final)** — re-runs all three models on the same eval set → `comparison.json`.

### Live monitoring

Open `https://wandb.ai/<your-entity>/medqa-llama3.1-8b` — you will see 3 runs:
- `sft-<ts>` — SFT loss / eval loss / grad norm / LR.
- `sasr-<ts>` — SASR warm-up + adaptive phase, with `sasr/phase` (0=SFT, 1=GRPO), `sasr/p_t`, `sasr/G_warmup`, GRPO reward/KL/surrogate.
- `eval-<ts>` — comparison tables for the 3 models.

---

## 3. Files produced

```
/workspace/models/medqa-sft/        # LoRA adapters + sft_meta.json
/workspace/models/medqa-sasr/       # LoRA adapters + sasr_meta.json
/workspace/outputs/baseline_eval.json
/workspace/outputs/sft_eval.json
/workspace/outputs/sasr_eval.json
/workspace/outputs/comparison.json  # ← the deliverable table
/workspace/outputs/qualitative_samples.md  # 20 triplets per model, side by side
/workspace/logs/{00..03}_*.log      # console logs of each stage
```

The `comparison.json` structure:
```json
{
  "macro": {
    "bertscore_f1": {"baseline": 0.68, "sft": 0.81, "sasr": 0.83},
    "rouge_l":      {"baseline": 0.10, "sft": 0.24, "sasr": 0.26},
    ...
  },
  "per_language": {
    "en":     {"bertscore_f1": {...}, ...},
    "fr":     {...},
    "darija": {...}
  }
}
```

---

## 4. Publishing the adapters to HuggingFace (BRAIN HEALTH M5 deliverable)

```bash
python3 push_to_hub.py --username Williamsanderson --prefix MedQA-Llama3.1-8B
# creates and uploads:
#   Williamsanderson/MedQA-Llama3.1-8B-SFT
#   Williamsanderson/MedQA-Llama3.1-8B-SASR
# with auto-generated model cards including training recipe + metrics.
```

Use `--private` to keep them private initially and flip later from the HF UI.

---

## 5. Cost estimate (real-world)

| Stage | Duration (A40) | Duration (L40S) |
|---|---:|---:|
| Setup + model download | 15 min | 15 min |
| Baseline eval (300 samples) | 25 min | 12 min |
| **SFT 1 epoch** (35K × QLoRA) | ~4 h | ~1.8 h |
| **SASR 1 epoch** (warmup + adaptive) | ~6-7 h | ~3 h |
| Final eval (3 models) | ~1 h | ~35 min |
| **Total** | ~12 h | ~5.5 h |
| **$ estimate (community)** | **~$4.70** | **~$4.80** |

L40S ties A40 on cost while being **2× faster**. Pick L40S.

For full safety margin, budget **$6-8 on L40S** or **$5-6 on A40**.

---

## 6. Why QLoRA (not full LoRA)?

For Llama-3.1-8B on a single 48 GB GPU:

| Method | Activation + optimizer memory | Effective batch | Quality loss |
|---|---:|---:|---|
| Full fine-tuning (bf16) | ~60 GB → **OOM** | — | — |
| Full LoRA (bf16) | ~36 GB | 4-8 | None |
| **QLoRA (4-bit + LoRA)** | ~14 GB | 16-32 | Negligible (<0.5% on most benchmarks — cf. Dettmers et al. 2023) |

**QLoRA is the right choice** because:
1. It fits with **3× more headroom** for longer sequences and bigger effective batches.
2. The quality gap vs full LoRA is within noise for SFT on a 35K dataset.
3. GRPO rollouts during SASR need free VRAM to generate 4 completions per step — QLoRA frees that.

The only case where full LoRA would win is if you had to push past 2048 max_len or you wanted zero-risk numerical fidelity; neither applies here.

---

## 7. SASR verification vs the paper

The paper (Chen et al., arXiv:2505.13026) specifies:

| Paper (Sec. 3 + Algo 1) | Our implementation | OK? |
|---|---|---|
| Warm-up: pure SFT for W steps, record `G_warmup = ‖∇L_SFT‖` | `SASR_WARMUP_STEPS=200` then `current_grad_norm(policy)` | ✓ |
| Eq. 12: `p_t = ‖∇L_SFT_t‖ / (‖∇L_SFT_t‖ + γ·‖∇L_SFT_0‖)` | exact formula in `train_sasr.py` line `p = g_last_sft / (g_last_sft + γ·g_warmup)` | ✓ |
| Algo 1: `α ~ U(0,1); if α<p → SFT else → GRPO` | `is_sft = alpha < p` | ✓ |
| GRPO: sample G outputs, group-relative advantage `(r - mean)/(std+ε)` | `rewards = [reward(c, gold) for c in G completions]; adv = (r - r.mean())/(r.std()+ε)` | ✓ |
| Clipped surrogate: `-min(ρÂ, clip(ρ,1±ε)Â)` + β·KL(π‖π_ref) | `surrogate + β·KL`, with π_ref = frozen SFT-warm copy | ✓ |

Simplifications (documented in the model card):
- **Reward = ROUGE-L vs gold** (instead of a trained reward model). For medical free-form answers there is no standard reward model; ROUGE-L is the cheapest defensible proxy. Can be swapped for BERTScore or a small reward model without changing the SASR scaffolding.
- **Single update per rollout** (so ratio ρ=1 and clipping is vestigial). Multi-update variants are a known simplification trade-off in on-policy PPO-style training.

---

## 8. Troubleshooting

**OOM during SFT** → drop `PER_DEVICE_BATCH` from 4 → 2 in `config.py`, bump `GRAD_ACCUM_STEPS` from 4 → 8 to keep effective batch 16.

**Llama-3.1 download fails** → you must have accepted the license on https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct with the account whose token you use.

**wandb runs don't show up** → `wandb login $WANDB_API_KEY` must run BEFORE `launch.sh`. Check `~/.netrc` has the token.

**SASR GRPO very slow (>30 s/step)** → reduce `SASR_G_SAMPLES` from 4 → 2, or `SASR_GEN_MAX_NEW` from 256 → 128.

**Pod goes to sleep** → use `tmux new -s train` then `bash launch.sh` so you can detach (`Ctrl+b d`) and reattach from the web terminal.

---

## 9. Next steps after pipeline runs

1. Open `comparison.json` — sanity-check that **SFT > baseline** on at least BERTScore and ROUGE-L (otherwise training didn't work).
2. Open `qualitative_samples.md` — read 20 answers across models. Check for hallucinations, answer length drift, language confusion.
3. Push to HF Hub with `push_to_hub.py`.
4. Write the M5 evaluation report (10-15 pages) using the tables from `comparison.json` and examples from `qualitative_samples.md`.

# BRAIN HEALTH — 4-way LLM benchmark on RunPod L40S

End-to-end pipeline producing **3 fine-tuned Llama-3.1-8B-Instruct adapters**
and a paper-style comparison against the un-tuned baseline:

| # | Adapter | Dataset | Method | +Dorosz KG |
|---|---|---|---|:---:|
| 0 | (baseline) | — | none | — |
| 1 | **MedQA-Llama3.1-8B-SFT-Big**   | `Williamsanderson/MedQA-Darija-MultiLingual` (100K, trilingual) | QLoRA SFT          | ✓ |
| 2 | **MedQA-Llama3.1-8B-SASR-Big**  | same as #1                                                       | SFT-warm + GRPO (composite reward) | ✓ |
| 3 | **MedQA-Llama3.1-8B-SFT-Small** | `BrainHealthAI/MedQA_mutilangual` (68K + test, ctx+question)     | QLoRA SFT          | ✗ |

All adapters teach the model to answer with `<think>…</think><answer>…</answer>` and to **reply in the same language as the question** (EN / FR / Darija). Held-out trilingual eval (300 samples × 3 languages).

---

## 0. Pre-flight checklist (do this LOCALLY before booting the pod)

1. **HuggingFace** — make sure your token has accepted the Llama-3.1 license:
   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct → Click "Accept the license"
   - Get a write-token at https://huggingface.co/settings/tokens
2. **Wandb API key** — https://wandb.ai/authorize (paste once, copy)
3. The `brain_health_training/` folder (this folder) must be uploadable as a single zip — about 5 MB total (the bundled `dorosz_ckg.json` is the bulk).

---

## 1. Boot the RunPod L40S pod

1. Go to https://www.runpod.io → **Pods → Deploy**
2. Pick:
   - **GPU:** L40S (48 GB) — `Community Cloud` priced ~$0.86/h
   - **Pod template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - **Container disk:** ≥ 30 GB
   - **Volume:** ≥ 60 GB (Llama-3.1-8B weights = 16 GB, then datasets + 3 adapter dirs)
3. Open the **Web Terminal**.

---

## 2. Upload this folder to the pod

From the Web Terminal:

```bash
cd /workspace
mkdir -p scripts && cd scripts
# Easiest: drop the entire brain_health_training/ folder contents here
# via the RunPod file browser (left sidebar). Make sure dorosz_ckg.json
# lands in /workspace/scripts/ alongside the .py files.
```

After upload, verify:

```bash
ls /workspace/scripts/
# should list: config.py data.py dorosz_kg.py dorosz_ckg.json evaluate.py
#              launch.sh plot_results.py push_to_hub.py setup_runpod.sh
#              train_sasr.py train_sft.py LAUNCH_INSTRUCTIONS.md
```

---

## 3. One-time setup

```bash
cd /workspace/scripts
chmod +x setup_runpod.sh launch.sh
bash setup_runpod.sh
```

Expected output ends with:

```
GPU  NVIDIA L40S  (44.5 GB)
```

Then authenticate (paste your tokens):

```bash
export HF_TOKEN="hf_xxx"
export WANDB_API_KEY="xxx"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_API_KEY
```

---

## 4. Smoke test (5 minutes — strongly recommended before the real run)

```bash
LIMIT=200 SAMPLES_PER_LANG=20 STOP_AT=2 bash launch.sh
```

This runs baseline eval + tiny SFT-Big + tiny SASR-Big on 200 samples, takes
about 25 min, costs ~$0.40, and confirms the whole pipeline works end-to-end.

If wandb shows two runs (`sft-big-…`, `sasr-big-…`) and `comparison.json`
contains numbers > 0, you're good to go.

---

## 5. Real run

```bash
tmux new -s train
bash launch.sh 2>&1 | tee /workspace/full_run.log
```

(Detach with `Ctrl+b` then `d`. Reattach with `tmux attach -t train`.)

### Expected timeline & cost (L40S @ ~$0.86/h)

| Stage | Approx duration | Cumulative cost |
|---|---:|---:|
| 0  Baseline eval (300/lang × 3) | ~30 min  | $0.45 |
| 1  SFT-Big (100K × 3 epochs)    | ~7-8 h   | $7.20 |
| 2  SASR-Big (warmup + adaptive) | ~12-14 h | $19.30 |
| 3  SFT-Small (68K × 3 epochs)   | ~5-6 h   | $24.45 |
| 4  Final eval (4 models)        | ~2 h     | $26.20 |
| 5  Plots                        | <1 min   | $26.20 |

**Total: ~28-32 h ≈ $25-28** of your $50 team budget. Plenty of margin.

### Budget guardrails

- `STOP_AT=N` lets you halt after stage N. E.g. `STOP_AT=2 bash launch.sh` runs
  baseline + SFT-Big + SASR-Big and stops — useful if you want to verify the
  big-data results before launching SFT-Small.
- Stop the pod immediately when launch.sh completes — RunPod bills as long as
  the pod is running, even if idle.

---

## 6. Live monitoring

While `launch.sh` runs you can watch:

- **Web Terminal**: console output is live in tmux.
- **Wandb dashboard**: https://wandb.ai/<your-entity>/brainhealth-llama3-8b
  - run `sft-big-…` — train/eval loss, gradient norm
  - run `sasr-big-…` — warmup curve, then `sasr/phase`, `sasr/p_t`,
    `grpo/reward_mean`, `grpo/format_mean`, `grpo/content_mean`, `grpo/kl`
  - run `sft-small-…` — same panel as sft-big
  - run `eval-…` — final comparison_macro table
- **GPU live**: in another terminal, `watch -n 5 nvidia-smi`

If `grpo/reward_mean` plateaus < 0.2 in early SASR steps, that's normal — the
format-and-content reward composite needs 1-2 thousand steps to start moving.
If it's still flat after 5K steps, lower `SASR_W_FORMAT` to 0.20 and bump
`SASR_W_CONTENT` to 0.80 in `config.py`, kill the run, and restart stage 2.

---

## 7. Outputs

Everything is on the pod under `/workspace/`:

```
models/
  medqa-sft-big/      LoRA adapters + sft_meta.json
  medqa-sasr-big/     LoRA adapters + sasr_meta.json
  medqa-sft-small/    LoRA adapters + sft_meta.json
outputs/
  baseline_eval.json
  sft_big_eval.json
  sasr_big_eval.json
  sft_small_eval.json
  comparison.json           ← the deliverable table (14 metrics × 4 models × 3 langs)
  qualitative_samples.md    ← side-by-side examples per model
  three_lang_demo.json      ← the 3 fixed demo prompts answered by all 4 models
  plots/
    fig_bleu_comparison.png
    fig_bertscore_comparison.png
    fig_rouge_l_comparison.png
    fig_token_f1_comparison.png
    fig_radar_macro.png
    fig_per_language_heatmap.png
    fig_full_table.png      ← paper-ready 14-row table
logs/
  00_baseline_eval.log
  01_sft_big_train.log
  02_sasr_big_train.log
  03_sft_small_train.log
  04_final_eval.log
  05_plots.log
```

---

## 8. Push the 3 adapters to HuggingFace

```bash
cd /workspace/scripts
python3 push_to_hub.py --username Williamsanderson --prefix MedQA-Llama3.1-8B
# creates and uploads:
#   Williamsanderson/MedQA-Llama3.1-8B-SFT-Big      (model card with metrics)
#   Williamsanderson/MedQA-Llama3.1-8B-SASR-Big     (+ comparison.json + plots/ + qualitative_samples.md)
#   Williamsanderson/MedQA-Llama3.1-8B-SFT-Small    (model card with metrics)
```

Add `--private` if you want them private until validation.

---

## 9. Download artefacts back to your laptop

Before stopping the pod, pull the small but irreplaceable artefacts:

```bash
# In the RunPod web file browser, navigate to:
#   /workspace/outputs/    → drag-download the whole folder (~10 MB)
#   /workspace/full_run.log
```

The model adapters are also on HuggingFace after step 8, so you don't have to
download those. (Each is ~700 MB.)

---

## 10. Stop the pod

When everything is uploaded:

> RunPod web UI → Pods → your pod → **Stop** (NOT Terminate, unless you don't
> want to keep the volume).

Stopping suspends billing of the GPU but keeps your volume so you can resume
later if you want to retrain.

---

## 11. After-run analysis

### What's automatically computed in stage 4

The final `evaluate.py` produces 3 layers of metrics, all aggregated in `comparison.json`:

| Layer | Keys in `comparison.json` | What it measures |
|---|---|---|
| **Lexical / semantic** | `macro`, `per_language`, `deltas_vs_baseline` | BLEU, GLEU, ROUGE-1/2/L, METEOR, token-P/R/F1, BS_P/R/F1, SB_CS, NASS — 14 metrics ×4 models × 3 langs |
| **Medical-grade**      | `medical_macro`, `medical_per_pathology` | Entity recall (drugs/symptoms/exams), hallucination rates per category, dosage hallucination count (safety alarm), Diagnostic Precision@3, citation rate, +13 pathology buckets |
| **Robustness** *(opt.)* | `robustness` | Paraphrase coefficient-of-variation on BLEU/BS_F1/ROUGE-L, ASR-noise Δ on the same metrics, per model |

### Enabling the robustness suite

Robustness is OFF by default (saves ~2h on L40S = ~$1.70). Turn it on via env var:

```bash
RUN_ROBUSTNESS=1 bash launch.sh        # full pipeline + stress tests
# OR re-run just stage 4 after the fact:
python3 evaluate.py --robustness --robustness-n 30
```

What the suite measures:
- **Paraphrase**: each prompt is reformulated 3 different ways (multilingual rules), and we compute the coefficient of variation (CV = std/mean) of BLEU/ROUGE-L/BS_F1 across the variants. **Lower CV = more stable model.**
- **ASR noise**: the prompt is perturbed with realistic transcription errors (char swaps/drops, FR/EN homophones), then re-evaluated. We compute the metric drop vs the clean version. **Lower drop = more robust to ASR upstream.**

### Human-evaluation CSV (medical professional review)

Stage 4 also produces `outputs/human_eval_template.csv` (40 anonymised Q&A triplets, 4 models per row labelled A/B/C/D randomly). Send it to a doctor / med student. They fill scores 1-5 on:
- clinical correctness
- completeness
- safety
- reasoning quality
- overall

When they return the filled CSV(s):
```bash
python3 human_eval_analyze.py reviewer_1.csv reviewer_2.csv
# → outputs/human_eval_aggregated.json + human_eval_kappa.json (Cohen κ)
```
Aim for κ ≥ 0.6 between reviewers to consider the human evaluation reliable.

### Replay analyses locally

After downloading `comparison.json` to your laptop:

```python
import json
comp = json.load(open("comparison.json"))

# Lexical metric — macro BS_F1 per model
for k, v in comp["macro"]["BS_F1"].items():
    print(f"{k:<10} {v:.3f}")

# Medical safety alarm — count of dosage-hallucination samples
for k, v in comp["medical_macro"]["n_dosage_hallu_samples"].items():
    print(f"{k:<10} {v} samples with dosage hallucination")

# Pathology breakdown — entity recall on cardiovascular cases
for k, v in comp["medical_per_pathology"].get("cardiovascular", {}).items():
    print(f"{k:<10} n={v.get('n')} recall={v.get('entity_recall_macro'):.3f}")

# Robustness — paraphrase CV (lower = more stable)
if "robustness" in comp:
    for k, r in comp["robustness"].items():
        print(f"{k:<10} BS_F1 CV under paraphrase: {r['paraphrase']['BS_F1_cv']:.3f}")
```

---

## 12. Hypothesis to verify

The headline question is **whether SASR-Big > SFT-Big > SFT-Small > baseline**
on macro-avg BERTScore-F1 and token-F1 — and whether the extra `<think>` block
in SASR rollouts measurably improves clinical reasoning quality (look at
`qualitative_samples.md` for the "feel" + `plots/fig_radar_macro.png` for the
shape of improvement).

Expected delta-vs-baseline (rough, from analogous public benchmarks):

|              | BLEU  | ROUGE-L | BS_F1 |
|---|---:|---:|---:|
| baseline     | ~0.025 | ~0.245 | ~0.857 |
| SFT-Small    | +0.060 | +0.080 | +0.018 |
| **SFT-Big**  | +0.090 | +0.110 | +0.025 |
| **SASR-Big** | +0.115 | +0.135 | +0.034 |

If SASR-Big ≥ SFT-Big by a margin > 1× the std-dev across the 5 qualitative
samples per language, the GRPO composite-reward training added measurable
value over plain SFT — exactly the result we want to publish.

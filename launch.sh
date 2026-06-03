#!/bin/bash
# launch.sh — BRAIN HEALTH 4-way LLM benchmark.
#
# Stages (each is its own log + wandb run):
#   0  baseline eval               (~30 min)
#   1  SFT on big data             (~7-8 h)   100K × 3 epochs +Dorosz
#   2  SASR on big data            (~12-14 h) starts from SFT-Big checkpoint
#   3  SFT on small data           (~5-6 h)   68K × 3 epochs (no Dorosz)
#   4  Final eval (4 models)       (~2 h)
#   5  Generate paper-style plots  (<1 min)
#
# Total ~28-32 h on L40S → ~$25-28 with $0.86/h community pricing.
# Set STOP_AT="N" to halt after stage N (useful for budget control).
set -e

cd /workspace/scripts

echo ""
echo "##################################################################"
echo "# BRAIN HEALTH 4-way LLM benchmark                               #"
echo "# Llama-3.1-8B-Instruct  ×  {SFT_big, SASR_big, SFT_small}       #"
echo "##################################################################"
date

LIMIT=${LIMIT:-}
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  echo "Smoke-test mode: --limit $LIMIT"
  LIMIT_ARG="--limit $LIMIT"
fi
SAMPLES_PER_LANG=${SAMPLES_PER_LANG:-300}
STOP_AT=${STOP_AT:-5}
# Set RUN_ROBUSTNESS=1 to add ~30 min/model of paraphrase + ASR-noise stress tests
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-0}
ROBUSTNESS_ARG=""
if [ "$RUN_ROBUSTNESS" = "1" ]; then
  ROBUSTNESS_ARG="--robustness --robustness-n 30"
  echo "Robustness suite: ENABLED (+~2 h, ~$1.7 on L40S)"
fi

mkdir -p /workspace/logs

# ── 0. Baseline eval ─────────────────────────────────────────────────────
if [ "$STOP_AT" -ge 0 ]; then
  echo ""; echo ">>> [0/5] BASELINE evaluation (no fine-tuning)"
  python3 evaluate.py --skip-sft-big --skip-sasr-big --skip-sft-small --skip-demo \
      --samples-per-lang $SAMPLES_PER_LANG \
      2>&1 | tee /workspace/logs/00_baseline_eval.log
fi
[ "$STOP_AT" -le 0 ] && { echo "STOP_AT=$STOP_AT, exiting after baseline."; exit 0; }

# ── 1. SFT on big data ───────────────────────────────────────────────────
if [ "$STOP_AT" -ge 1 ]; then
  echo ""; echo ">>> [1/5] SFT on BIG data (MedQA-Darija-MultiLingual + Dorosz)"
  python3 train_sft.py --dataset big $LIMIT_ARG \
      2>&1 | tee /workspace/logs/01_sft_big_train.log
fi
[ "$STOP_AT" -le 1 ] && { echo "STOP_AT=$STOP_AT, exiting after SFT-big."; exit 0; }

# ── 2. SASR on big data ──────────────────────────────────────────────────
if [ "$STOP_AT" -ge 2 ]; then
  echo ""; echo ">>> [2/5] SASR on BIG data (composite reward + Dorosz)"
  python3 train_sasr.py $LIMIT_ARG --sft-path /workspace/models/medqa-sft-big \
      2>&1 | tee /workspace/logs/02_sasr_big_train.log
fi
[ "$STOP_AT" -le 2 ] && { echo "STOP_AT=$STOP_AT, exiting after SASR-big."; exit 0; }

# ── 3. SFT on small data ─────────────────────────────────────────────────
if [ "$STOP_AT" -ge 3 ]; then
  echo ""; echo ">>> [3/5] SFT on SMALL data (BrainHealthAI/MedQA_mutilangual, no KG)"
  python3 train_sft.py --dataset small $LIMIT_ARG \
      2>&1 | tee /workspace/logs/03_sft_small_train.log
fi
[ "$STOP_AT" -le 3 ] && { echo "STOP_AT=$STOP_AT, exiting after SFT-small."; exit 0; }

# ── 4. Final eval (all 4 models) ─────────────────────────────────────────
if [ "$STOP_AT" -ge 4 ]; then
  echo ""; echo ">>> [4/5] FINAL evaluation (4 models, +medical metrics, +robustness?)"
  python3 evaluate.py --samples-per-lang $SAMPLES_PER_LANG $ROBUSTNESS_ARG \
      2>&1 | tee /workspace/logs/04_final_eval.log
  # Build the human-eval CSV template (medical professional review)
  python3 human_eval_export.py --n-questions 40 \
      2>&1 | tee /workspace/logs/04b_human_eval_export.log
fi
[ "$STOP_AT" -le 4 ] && { echo "STOP_AT=$STOP_AT, exiting after eval."; exit 0; }

# ── 5. Plots ─────────────────────────────────────────────────────────────
if [ "$STOP_AT" -ge 5 ]; then
  echo ""; echo ">>> [5/5] Generating comparison plots"
  python3 plot_results.py 2>&1 | tee /workspace/logs/05_plots.log
fi

echo ""
echo "##################################################################"
echo "  Pipeline complete."
echo "  Logs:    /workspace/logs/"
echo "  Models:  /workspace/models/{medqa-sft-big, medqa-sasr-big, medqa-sft-small}"
echo "  Reports: /workspace/outputs/{baseline,sft_big,sasr_big,sft_small}_eval.json"
echo "           /workspace/outputs/comparison.json"
echo "           /workspace/outputs/qualitative_samples.md"
echo "           /workspace/outputs/three_lang_demo.json"
echo "           /workspace/outputs/plots/*.png"
echo "##################################################################"
date

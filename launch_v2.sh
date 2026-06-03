#!/usr/bin/env bash
# HELIX-FT v2 — End-to-end launcher (run from inside RunPod L40S pod).
#
# Stages: 0 setup → 1 merge SFT-Big → 2 Stage A (SFT 20K) → 3 Stage B (SASR 3K) →
#         4 Stage C (CoT 3K) → 5 evaluate → 6 push.
#
# Usage:
#   bash launch_v2.sh                     # run everything
#   START_AT=2 bash launch_v2.sh          # resume from Stage A
#   STOP_AT=2 bash launch_v2.sh           # stop after Stage A
#   SKIP_PUSH=1 bash launch_v2.sh         # run all but the final HF push
# pipefail is critical — without it, `tee` in run_stage's pipe swallows exit codes and
# the script merrily runs ALL stages even when earlier ones crash.
set -eo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
SCRIPTS=${SCRIPTS:-$WORKSPACE/scripts}     # this dir, uploaded to the pod
MODELS=$WORKSPACE/models
OUTPUTS=$WORKSPACE/outputs
LOGS=$WORKSPACE/logs
mkdir -p "$MODELS" "$OUTPUTS" "$LOGS"

# Force ALL stages (A SFT, B SASR, C CoT) into the same WandB project so the live
# dashboard shows one consolidated view, not 3 separate runs in unrelated projects.
export WANDB_PROJECT=${WANDB_PROJECT:-helix-v2}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-helix-v2-$(date +%Y%m%d-%H%M)}

START_AT=${START_AT:-0}
STOP_AT=${STOP_AT:-6}

run_stage() { # $1 = stage_idx, $2 = name, $3.. = command
    local idx=$1 name=$2; shift 2
    if (( idx < START_AT )); then echo "→ skip $idx $name (before START_AT=$START_AT)"; return; fi
    if (( idx > STOP_AT )); then echo "→ stop before $idx $name (STOP_AT=$STOP_AT)"; exit 0; fi
    echo ""; echo "=============================================="
    echo "STAGE $idx — $name   ($(date '+%H:%M:%S'))"
    echo "=============================================="
    "$@" 2>&1 | tee "$LOGS/${idx}_${name}.log"
}

# ── Stage 0: env check ────────────────────────────────────────────────────
run_stage 0 setup bash -c '
    nvidia-smi || { echo "No GPU"; exit 1; }
    # Pinned versions known to be compatible with torch 2.4.x (RunPod base image).
    # Do NOT widen to >=4.45 — transformers 4.50+ needs torch>=2.5 (device_mesh).
    # flash-attn is intentionally NOT auto-installed (slow compile + flaky). We use SDPA.
    pip install --quiet --upgrade \
        "transformers==4.46.3" "peft==0.13.2" "trl==0.11.4" \
        "bitsandbytes==0.43.3" "accelerate==0.34.2" \
        "datasets==2.21.0" "sentence-transformers" "rouge_score" "wandb" 2>&1 | tail -5
    python -c "import torch, transformers; print(f\"torch {torch.__version__} / transformers {transformers.__version__}\")"
'

# ── Stage 1: merge SFT-Big into Llama-3.1-8B-Instruct ────────────────────
run_stage 1 merge_sft_big python "$SCRIPTS/merge_sft_big.py" \
    --out "$MODELS/brainmed-base-v1"

# ── Stage 2: Continued SFT on MedUnified (20K, 2 epochs) ─────────────────
run_stage 2 stage_a_sft python "$SCRIPTS/train_stage_a_sft.py" \
    --base "$MODELS/brainmed-base-v1" \
    --out  "$MODELS/brainmed-sft-v2"

# ── Stage 3: SASR (3K, G=2, ~1000 steps) ─────────────────────────────────
run_stage 3 stage_b_sasr python "$SCRIPTS/train_stage_b_sasr.py" \
    --base    "$MODELS/brainmed-base-v1" \
    --adapter "$MODELS/brainmed-sft-v2" \
    --out     "$MODELS/brainmed-sasr-v2" \
    --kg      "$SCRIPTS/dorosz_ckg.json"

# ── Stage 4: CoT polish (3K, 1 epoch) ─────────────────────────────────────
run_stage 4 stage_c_cot python "$SCRIPTS/train_stage_c_cot.py" \
    --base    "$MODELS/brainmed-base-v1" \
    --adapter "$MODELS/brainmed-sasr-v2" \
    --out     "$MODELS/brainmed-cot-v2"

# ── Stage 5: evaluate (MCQ + generation) ─────────────────────────────────
run_stage 5 evaluate python "$SCRIPTS/evaluate_v2.py" \
    --base    "$MODELS/brainmed-base-v1" \
    --adapter "$MODELS/brainmed-cot-v2" \
    --out     "$OUTPUTS/eval_v2.json" \
    --mcq-limit 200 --gen-limit 100

# ── Stage 6: push final adapter to HF ────────────────────────────────────
if [[ -z "$SKIP_PUSH" ]]; then
    run_stage 6 push python -c "
import os
from huggingface_hub import HfApi, create_repo
tok = os.environ['HF_TOKEN']
repo = 'BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2'
api = HfApi(token=tok)
create_repo(repo, repo_type='model', private=True, token=tok, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path='$MODELS/brainmed-cot-v2', token=tok)
api.upload_file(path_or_fileobj='$OUTPUTS/eval_v2.json', path_in_repo='eval_v2.json',
                repo_id=repo, token=tok)
print(f'pushed -> https://huggingface.co/{repo} (PRIVATE)')
"
fi

echo ""
echo "=============================================="
echo "ALL STAGES COMPLETE — outputs in $OUTPUTS, models in $MODELS"
echo "=============================================="

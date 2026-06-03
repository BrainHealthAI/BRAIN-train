"""Shared configuration for the BRAIN HEALTH 4-way LLM benchmark.

Pipeline:
  1) Baseline   : Llama-3-8B-Instruct, no fine-tuning
  2) SFT-Big    : QLoRA SFT on Williamsanderson/MedQA-Darija-MultiLingual (100K, +Dorosz KG)
  3) SASR-Big   : Step-wise Adaptive SFT + GRPO on top of SFT-Big (composite reward, +Dorosz KG)
  4) SFT-Small  : QLoRA SFT on BrainHealthAI/MedQA_mutilangual (68K, no KG)

Hardware target: a single RunPod L40S (48 GB). Budget ~$25-30.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
WORKSPACE     = Path("/workspace")
CACHE_DIR     = WORKSPACE / "cache"
MODELS_DIR    = WORKSPACE / "models"
OUTPUTS_DIR   = WORKSPACE / "outputs"
LOGS_DIR      = WORKSPACE / "logs"
PLOTS_DIR     = OUTPUTS_DIR / "plots"
DOROSZ_KG     = WORKSPACE / "scripts" / "dorosz_ckg.json"  # uploaded with the scripts

SFT_BIG_OUT     = MODELS_DIR / "medqa-sft-big"
SASR_BIG_OUT    = MODELS_DIR / "medqa-sasr-big"
SFT_SMALL_OUT   = MODELS_DIR / "medqa-sft-small"

BASELINE_EVAL   = OUTPUTS_DIR / "baseline_eval.json"
SFT_BIG_EVAL    = OUTPUTS_DIR / "sft_big_eval.json"
SASR_BIG_EVAL   = OUTPUTS_DIR / "sasr_big_eval.json"
SFT_SMALL_EVAL  = OUTPUTS_DIR / "sft_small_eval.json"
COMPARISON_OUT  = OUTPUTS_DIR / "comparison.json"

for d in [CACHE_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────
# Llama-3.1-8B-Instruct: 128K context, improved multilingual + instruction following
# vs Llama-3-8B (8K context). Same architecture, same param count.
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ── Datasets ──────────────────────────────────────────────────────────────
# A. The big trilingual one (267K rows). We sample BIG_LIMIT for cost reasons.
# 50K is the sweet spot: keeps the trilingual diversity, still trains in ~18h
# on L40S, leaves budget for SASR + SFT-Small. Stratified by specialty_id ⇒
# each of the ~57 trilingual-complete specialties gets ~877 samples.
HF_DATASET_BIG       = "Williamsanderson/MedQA-Darija-MultiLingual"
HF_DATASET_BIG_CONF  = "default"
BIG_LIMIT            = 50_000
STRATIFIED_BIG       = True

# B. The smaller, single-language Q&A with context.
# Cap to 20K to keep training time comparable to BIG (≈ same total compute) and
# fit the $50 team budget. The full 76K split would add ~$15 just for SFT-Small.
HF_DATASET_SMALL     = "BrainHealthAI/MedQA_mutilangual"
HF_DATASET_SMALL_CONF = "default"
SMALL_LIMIT          = 20_000

# Trilingual config for the big dataset:
LANGS = ["en", "fr", "darija"]

# ── QLoRA ─────────────────────────────────────────────────────────────────
LORA_R              = 64
LORA_ALPHA          = 128
LORA_DROPOUT        = 0.1
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# ── Training ──────────────────────────────────────────────────────────────
MAX_SEQ_LEN        = 2048
PER_DEVICE_BATCH   = 2          # user-requested; safe on L40S 48 GB w/ 8B QLoRA
GRAD_ACCUM_STEPS   = 8          # effective batch 16
LEARNING_RATE_SFT  = 2e-4
LEARNING_RATE_SASR = 5e-5       # lower for the RL-mixed phase
NUM_EPOCHS         = 3          # 5 would blow the $50 team budget
WARMUP_RATIO       = 0.03
LOGGING_STEPS      = 25
EVAL_STEPS         = 250
SAVE_STEPS         = 250
SAVE_TOTAL_LIMIT   = 2          # disk hygiene
MAX_GRAD_NORM      = 1.0
EARLY_STOP_PATIENCE = 3         # eval_loss flat for 3*EVAL_STEPS = stop

# ── SASR (paper Sec. 3.3 + reward design from CVD-thinking style) ─────────
SASR_WARMUP_STEPS  = 200
SASR_GAMMA         = 1.0
SASR_G_SAMPLES     = 4          # rollouts per prompt
SASR_BETA          = 0.02       # KL to SFT-warm reference
SASR_CLIP_EPS      = 0.2
SASR_GEN_MAX_NEW   = 320        # enough for <think>...</think><answer>...</answer>
SASR_GEN_TEMP      = 0.8

# Composite reward weights — see reward_composite() in train_sasr.py
SASR_W_FORMAT      = 0.30
SASR_W_CONTENT     = 0.70

# Adaptive-phase budget. Smoke-measured cost ≈ 62.5s/step on average
# (mix SFT@25s + GRPO@100s). 1500 steps ≈ 26h ≈ $22; 2000 ≈ 35h ≈ $30.
# Paper SASR uses 1500-3000 steps. 1500 is the sweet spot for the $50 team budget.
SASR_ADAPTIVE_STEPS_OVERRIDE = 1500

# ── Reasoning tags ────────────────────────────────────────────────────────
THINK_OPEN   = "<think>"
THINK_CLOSE  = "</think>"
ANSWER_OPEN  = "<answer>"
ANSWER_CLOSE = "</answer>"

# ── Evaluation ────────────────────────────────────────────────────────────
# Held-out per language. 300 × 3 = 900 samples per model × 4 models = 3600 generations.
EVAL_SAMPLES_PER_LANG = 300
EVAL_MAX_NEW_TOK      = 512
EVAL_BATCH_SIZE       = 4

# Sentence-transformer for SB_CS / NASS (multilingual)
SENT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ── Wandb ─────────────────────────────────────────────────────────────────
WANDB_PROJECT = "brainhealth-llama3-8b"
WANDB_ENTITY  = None

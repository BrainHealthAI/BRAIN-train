# BRAIN-train · Training_v2 — HELIX-FT v2

Second-generation medical LLM fine-tuning pipeline for **BrainHealthAI**. A 3-stage
QLoRA pipeline that takes `Llama-3.1-8B-Instruct`, merges the first-gen SFT-Big
adapter, then continues training on a curated multilingual (FR / Darija / EN) mix.

> Other branch: [`Training_v1`](../../tree/Training_v1) — the first-gen SFT-Big training (`runpod_training`).

---

## Pipeline (3 stages)

| Stage | Script | What it does |
|-------|--------|--------------|
| **1 — merge** | `merge_sft_big.py` | Merge SFT-Big LoRA into `Llama-3.1-8B-Instruct` → `BrainMed-Base-v1` |
| **A — SFT** | `train_stage_a_sft.py` | Continued QLoRA SFT on `BrainHealthAI/MedUnified` (~20K, 2 epochs) |
| **B — SASR** | `train_stage_b_sasr.py` + `sasr_reward.py` | Semantic-Aware Self-Refinement (reward-guided, ~3K) |
| **C — CoT** | `train_stage_c_cot.py` | Chain-of-Thought polish (~3K, 1 epoch) → final adapter |
| **eval** | `evaluate_v2.py`, `final_eval.py`, `reeval_loglikelihood.py` | MCQ benchmarks + generation metrics |
| **plot** | `plot_results.py` | Comparison plots from `eval_v2.json` |

Hyperparameters are **inline in each script** (not a shared `config.py`) — each stage is
self-documenting via its module docstring. Orchestration + paths are passed by
`launch_v2.sh` through CLI flags and env vars (`START_AT`, `STOP_AT`, `WANDB_PROJECT`).

## Quick start (RunPod L40S 48 GB)

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx          # must have accepted the Llama-3.1 license
export WANDB_API_KEY=xxx        # optional (logging)

bash launch_v2.sh               # run all stages 0→6
# resume / partial:
START_AT=2 bash launch_v2.sh    # resume from Stage A
STOP_AT=4 bash launch_v2.sh     # stop after Stage C
SKIP_PUSH=1 bash launch_v2.sh   # run all but final HF push
```

Recommended pod image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
(container ≥ 30 GB, volume ≥ 60 GB).

## Configuration (locked)

QLoRA: `r=64, alpha=128, dropout=0.1`, targets `q,k,v,o,gate,up,down`; 4-bit NF4 +
double-quant; bf16; sequence packing; `paged_adamw_8bit`; max_seq_len 2048; effective
batch 16 (2 × grad-accum 8); cosine LR with 5% warmup; SDPA attention.

## Evaluation

- **MCQ benchmarks**: MedQA-USMLE, MedMCQA, PubMedQA, MMLU (clinical_knowledge /
  medical_genetics / professional_medicine / college_medicine), and our
  `BrainHealthAI/MedQA-Darija-MCQ-500`.
- **Generation** (held-out `MedUnified` eval, per language): ROUGE-L, SB_CS,
  BLEU / METEOR / BERTScore-F1.

Latest MCQ results (`eval_v2.json`, limit 200):

| Benchmark | Accuracy | n |
|-----------|---------:|--:|
| MedQA-USMLE | 52.50% | 200 |
| MedMCQA | 41.50% | 200 |
| PubMedQA | 53.50% | 200 |
| MMLU-clinical_knowledge | 61.00% | 200 |
| MMLU-medical_genetics | 67.00% | 100 |
| MMLU-professional_medicine | 54.00% | 200 |
| MMLU-college_medicine | 57.23% | 173 |

## Artifacts

- Final adapter: [`BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2`](https://huggingface.co/BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2) (LoRA on Llama-3.1-8B-Instruct)
- Eval set: [`BrainHealthAI/MedUnified`](https://huggingface.co/datasets/BrainHealthAI/MedUnified) (`eval.parquet`, 495 held-out, trilingual)

> ⚠️ Research / educational use. Not a medical device; outputs require review by a
> qualified health professional.

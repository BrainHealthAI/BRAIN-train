# HELIX-FT v2 — RunPod launch guide

End-to-end continued training of `BrainHealthAI/MedQA-Llama3.1-8B-SFT-Big` on the new
**MedUnified** mix (20K, trilingual) → SASR (3K) → CoT (3K). Designed for a single L40S,
$13–16 budget.

## 0. Pre-flight (local, free)

1. `BrainHealthAI/MedUnified` pushed (private). If not yet pushed (token issues), upload
   `data_medunified/{train,eval}.parquet` directly to the pod under `/workspace/data/`
   and `data_v2.py` will fall back to local paths (TODO: add the fallback if needed).
2. `BrainHealthAI/MedQA-Darija-MCQ-500` pushed (public) — needed for the Darija eval.
3. Llama-3.1-8B-Instruct license accepted on your HF account.
4. WandB account ready (optional, set `WANDB_API_KEY`).

## 1. Boot the L40S pod

- GPU: **L40S 48 GB** (Community Cloud, ~$0.86/h)
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Container disk: ≥ 30 GB · Volume: ≥ 60 GB

## 2. Upload this folder + the dorosz KG

From the RunPod web terminal:
```bash
mkdir -p /workspace/scripts && cd /workspace/scripts
# Upload via the RunPod file browser (left sidebar):
#   - the entire training_v2/ folder content (this directory)
#   - dorosz_ckg.json  (anti-hallucination vocab for Stage B)
# AND ALSO upload the brain_health_training/ folder so sasr_reward.py is importable
ls
# should show: data_v2.py, merge_sft_big.py, train_stage_a_sft.py,
#              train_stage_b_sasr.py, train_stage_c_cot.py, evaluate_v2.py,
#              launch_v2.sh, dorosz_ckg.json, sasr_reward.py
```

## 3. Auth + launch (with tmux so you can detach safely)

```bash
export HF_TOKEN="hf_xxx"       # write token, with access to BrainHealthAI org
export WANDB_API_KEY="xxx"     # recommended — see §6 for what you'll see
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_API_KEY

cd /workspace/scripts
chmod +x launch_v2.sh

# Start a tmux session so the training survives if you disconnect:
tmux new -s train
bash launch_v2.sh 2>&1 | tee /workspace/full_run.log
# Detach: press Ctrl+B then D  (training keeps running)
# Reattach later: tmux attach -t train
```

## 4. Expected timeline + cost (L40S @ ~$0.86/h)

| Stage | Step | ~Duration | Cost |
|---|---|---:|---:|
| 0 | setup | 0.3 h | $0.26 |
| 1 | merge SFT-Big → BrainMed-Base-v1 | 0.3 h | $0.26 |
| 2 | Continued SFT 20K (2 epochs) | 5 h | $4.30 |
| 3 | SASR (3K, G=2, ~1100 steps) | 5 h | $4.30 |
| 4 | CoT polish (3K, 1 epoch) | 1.2 h | $1.03 |
| 5 | Evaluate (MCQ + gen on 5+ benchmarks) | 4 h | $3.44 |
| 6 | Push final adapter + eval json | 0.5 h | $0.43 |
| **Total** | | **~16 h** | **~$14** |

## 5. Outputs

- `/workspace/models/brainmed-cot-v2/` — the final LoRA adapter (~700 MB)
- `/workspace/outputs/eval_v2.json` — MCQ accuracy per benchmark + generation metrics per
  language (rouge_l, sb_cs)
- `/workspace/logs/*.log` — per-stage logs
- HF: `BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2` (private)

## 6. Live monitoring — what you see after `tmux detach`

The training is fully independent of your tmux session. After Ctrl+B D you can close the
RunPod terminal entirely and the run continues. Three ways to monitor remotely:

- **WandB dashboard (best, browser, no SSH needed)** — `https://wandb.ai/<your-entity>/helix-v2`
  - Stage A (`helix-v2-stageA-sft`): `train/loss`, `eval/loss`, `train/learning_rate`, `train/grad_norm`
  - Stage B (`stage-b-sasr-<ts>`): `sasr/reward_mean`, `sasr/format_mean`, `sasr/content_mean`,
    `sasr/hallu_mean`, `sasr/kl`, `sasr/p_t` (the SFT/GRPO mix probability), `sasr/loss`
  - Stage C (`helix-v2-stageC-cot`): same as Stage A
  - All 3 grouped under `WANDB_RUN_GROUP=helix-v2-<date>` so you see them as one experiment
- **`/workspace/full_run.log`** — full stdout from the `tee` redirect, downloadable via the
  RunPod file browser
- **`/workspace/logs/{0_setup,1_merge_sft_big,…,6_push}.log`** — per-stage logs (tee'd by
  `launch_v2.sh` for easier debugging)
- **GPU**: reattach with `tmux attach -t train`, open a 2nd pane (Ctrl+B then %), run
  `watch -n 5 nvidia-smi`. Detach again (Ctrl+B D) — `nvidia-smi` keeps watching too.

**Note**: WandB logging is conditional on `WANDB_API_KEY` being set BEFORE `launch_v2.sh`
runs. If you forget, no error but no dashboard either — only the local logs.

## 7. Failure recovery

- `STOP_AT=2 bash launch_v2.sh` → halt after Stage A (verify before paying for B)
- `START_AT=3 bash launch_v2.sh` → resume from Stage B if a stage crashed (the previous
  adapters are persisted in `/workspace/models/`)
- `SKIP_PUSH=1 bash launch_v2.sh` → run training+eval without HF push (useful in dev)

## 8. Decision gates

| Stage | Go signal | If failing |
|---|---|---|
| A | eval_loss < 0.75, grad_norm stable | Re-curate the 20K (likely too noisy) |
| B | `sasr/reward_mean` rises > 0.05 in first 200 steps | Reduce LR to 2e-6, increase warmup to 200 |
| C | eval_loss decreases | Keep Stage B model as final |

## 9. Honest performance expectations

- vs `MedQA-Llama3.1-8B-SFT-Big`: **+3 to +5 points SB_CS** on internal eval
- vs `Llama-3.1-8B-Instruct` baseline: **+8 to +15 points** on MedQA-USMLE accuracy
- vs **Meditron3-8B** on English: still behind (they have 50B+ medical tokens of training)
- vs **all open 8B medical models** on Darija: SOTA-accessible (no real competitor)

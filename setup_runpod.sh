#!/bin/bash
# setup_runpod.sh — One-time setup on a fresh RunPod L40S pod.
# Recommended pod image:
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Container disk ≥ 30 GB, Volume ≥ 60 GB.
set -e

echo "=== System packages ==="
apt-get update -qq
apt-get install -y -qq git-lfs vim htop tmux wget curl

echo "=== Python packages ==="
pip install --upgrade pip setuptools wheel

# Core ML stack
pip install --no-cache-dir \
  "torch>=2.3.0" torchaudio \
  "transformers>=4.45.0" \
  "accelerate>=0.34.0" \
  "peft>=0.12.0" \
  "bitsandbytes>=0.43.0" \
  "trl>=0.11.0" \
  "datasets>=2.20.0" \
  "huggingface_hub>=0.24.0" \
  sentencepiece protobuf safetensors

# Metrics
pip install --no-cache-dir \
  bert-score rouge-score sacrebleu nltk evaluate \
  sentence-transformers \
  scikit-learn pandas numpy matplotlib

# Tracking
pip install --no-cache-dir wandb

# Multilingual / Darija helpers
pip install --no-cache-dir sacremoses

# NLTK data (punkt / wordnet for METEOR + BLEU + GLEU)
python -c "import nltk; [nltk.download(p, quiet=True) for p in \
  ['punkt','punkt_tab','wordnet','omw-1.4']]"

mkdir -p /workspace/cache /workspace/models /workspace/logs /workspace/outputs

echo ""
echo "=== Environment check ==="
python - <<'PY'
import torch, transformers, peft, trl, bitsandbytes, datasets, sentence_transformers
print(f"  torch                {torch.__version__}  (cuda={torch.cuda.is_available()})")
print(f"  transformers         {transformers.__version__}")
print(f"  peft                 {peft.__version__}")
print(f"  trl                  {trl.__version__}")
print(f"  bitsandbytes         {bitsandbytes.__version__}")
print(f"  datasets             {datasets.__version__}")
print(f"  sentence-transformers {sentence_transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU                  {p.name}  ({p.total_memory/1024**3:.1f} GB)")
PY

echo ""
echo "=== Authentication reminder ==="
echo "Before running launch.sh, you must:"
echo "  export HF_TOKEN=hf_xxx        (must have accepted Llama-3.1 license)"
echo "  export WANDB_API_KEY=xxx"
echo "  huggingface-cli login --token \$HF_TOKEN --add-to-git-credential"
echo "  wandb login \$WANDB_API_KEY"
echo ""
echo "Then:  tmux new -s train  &&  bash launch.sh"

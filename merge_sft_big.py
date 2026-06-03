"""
Merge BrainHealthAI/MedQA-Llama3.1-8B-SFT-Big LoRA adapter into Llama-3.1-8B-Instruct
to produce the new base `BrainMed-Base-v1` that stages A/B/C will train on top of.

Why merge vs LoRA-on-LoRA: chaining LoRAs creates artefacts (the second LoRA's gradients
flow through an already-perturbed forward); merging gives us a clean dense base. Same
final-model size (only one adapter to push later), simpler debugging, no stacking weirdness.

Run on the pod (15-20 min on L40S, mostly download time):
    python merge_sft_big.py --out /workspace/models/brainmed-base-v1
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("merge_sft_big")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "BrainHealthAI/MedQA-Llama3.1-8B-SFT-Big"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for the merged dense model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    log.info("Loading base model %s ...", BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map="auto",
        cache_dir=args.cache_dir, attn_implementation="sdpa",
    )
    base.config.use_cache = False  # only relevant for training; harmless at merge

    log.info("Loading adapter %s ...", ADAPTER)
    model = PeftModel.from_pretrained(base, ADAPTER, cache_dir=args.cache_dir)

    log.info("Merging adapter into base weights ...")
    merged = model.merge_and_unload()

    log.info("Saving merged model -> %s", args.out)
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")

    # Save the tokenizer too (use the one from the adapter — it may have added tokens)
    tok = AutoTokenizer.from_pretrained(ADAPTER, cache_dir=args.cache_dir)
    tok.save_pretrained(args.out)

    log.info("DONE. Merged BrainMed-Base-v1 at %s", args.out)


if __name__ == "__main__":
    main()

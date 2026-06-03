"""
Stage A — Continued QLoRA SFT on `BrainHealthAI/MedUnified` (20K).

Base: `BrainMed-Base-v1` (Llama-3.1-8B-Instruct + SFT-Big merged via merge_sft_big.py).
Adds a NEW LoRA on top, trained on the 20K curated multilingual mix.

Hyperparameters (locked, tuned for the $19 budget):
  r=64, alpha=128, dropout=0.1, target_modules=[q,k,v,o,gate,up,down]
  LR=5e-5, cosine, warmup 5%, 2 epochs, batch eff 16 (2 x 8), max_seq_len=2048
  paged_adamw_8bit, bf16, sequence packing, flash-attention-2
  eval_steps=200, save best on eval_loss (patience 3)

Run on the pod:
  python train_stage_a_sft.py --base /workspace/models/brainmed-base-v1 \\
                              --out  /workspace/models/brainmed-sft-v2
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_v2 import load_medunified_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to BrainMed-Base-v1 (merged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    log.info("Loading datasets...")
    train_ds, eval_ds = load_medunified_sft()
    log.info("  train=%d, eval=%d", len(train_ds), len(eval_ds))

    log.info("Loading base + tokenizer...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=args.cache_dir)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    lora = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.1, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        bf16=True, tf32=True,
        logging_steps=25,
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=args.max_seq_len,
        packing=True,
        dataset_text_field="text",
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else "none",
        run_name=f"helix-v2-stageA-sft",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model, args=cfg, tokenizer=tok,
        train_dataset=train_ds, eval_dataset=eval_ds,
    )
    log.info("Starting Stage A training...")
    trainer.train()
    log.info("Saving best adapter -> %s", args.out)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    log.info("Stage A DONE.")


if __name__ == "__main__":
    main()

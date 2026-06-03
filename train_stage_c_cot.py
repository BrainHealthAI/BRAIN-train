"""
Stage C — Final polish QLoRA SFT on BrainMedCoT (3K with <think>...</think> structure).

Starts from the LoRA produced by Stage B (SASR). Trains for 1 epoch only — this stage's
goal is to lock the format (<think>{reasoning}</think>{answer}) on top of an already-
reasoning-improved model from SASR, not to teach broad new knowledge.

Run on the pod:
  python train_stage_c_cot.py --base /workspace/models/brainmed-base-v1 \\
                              --adapter /workspace/models/brainmed-sasr-v2 \\
                              --out /workspace/models/brainmed-cot-v2
"""
from __future__ import annotations

import argparse, logging, os, sys
import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(__file__))
from data_v2 import load_brainmedcot_for_cot_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_c")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="BrainMed-Base-v1 merged path")
    ap.add_argument("--adapter", required=True, help="Path to Stage B (SASR) LoRA adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    log.info("Loading CoT subset (target %d)...", args.n)
    train_ds, eval_ds = load_brainmedcot_for_cot_stage(n=args.n)
    log.info("  train=%d, eval=%d", len(train_ds), len(eval_ds))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    log.info("Loading base + Stage B adapter (trainable)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.print_trainable_parameters()

    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=args.cache_dir)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=1.0,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=3e-5, lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_grad_norm=1.0, optim="paged_adamw_8bit", bf16=True, tf32=True,
        logging_steps=20,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=2048, packing=True,
        dataset_text_field="text",
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else "none",
        run_name="helix-v2-stageC-cot",
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=cfg, tokenizer=tok,
                          train_dataset=train_ds, eval_dataset=eval_ds)
    log.info("Starting Stage C (CoT) training...")
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    log.info("Stage C DONE -> %s", args.out)


if __name__ == "__main__":
    main()

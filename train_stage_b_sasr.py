"""
Stage B — SASR (step-wise adaptive SFT + GRPO) on a 3K MedUnified subset.

Continues training the LoRA adapter produced by Stage A. Reward = corrected composite from
brain_health_training/sasr_reward.py (SB_CS semantic content + format + anti-hallucination
against Dorosz drugs), NOT the original ROUGE-L that caused the first SASR run to plateau.

PRUDENT scope (locked with user):
  - 3000 samples (stratified 40% EN / 30% FR / 30% darija)
  - G=2 rollouts per prompt
  - ~1000 SASR steps after 100 SFT-warmup steps
  - LR 5e-6 (RL is sensitive), KL beta 0.02

Run on the pod:
  python train_stage_b_sasr.py --base /workspace/models/brainmed-base-v1 \\
                                --adapter /workspace/models/brainmed-sft-v2 \\
                                --out /workspace/models/brainmed-sasr-v2 \\
                                --kg /workspace/scripts/dorosz_ckg.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from data_v2 import load_medunified_sasr_subset
import sasr_reward  # bundled in training_v2/

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_b")

WARMUP_STEPS = 100
SASR_STEPS = 1000
G_SAMPLES = 2
CLIP_EPS = 0.2
KL_BETA = 0.02
LR = 5e-6
MAX_SEQ_LEN = 2048
GEN_MAX_NEW = 320
GEN_TEMP = 0.8
GRAD_ACCUM = 8
PER_DEVICE_BATCH = 2
SEED = 42


def _load_kg_drug_vocab(kg_path: str) -> set[str]:
    if not kg_path or not os.path.isfile(kg_path):
        log.warning("Dorosz KG not found at %s — anti-hallu penalty will be disabled.", kg_path)
        return set()
    data = json.loads(open(kg_path, encoding="utf-8").read())
    return {n["label"].strip() for n in data.get("nodes", [])
            if n.get("type") == "drug" and isinstance(n.get("label"), str) and len(n["label"]) >= 4}


def _sft_step(policy, tokenizer, batch_texts):
    enc = tokenizer(batch_texts, padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt").to(policy.device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return policy(**enc, labels=labels).loss


def _logprobs_completion(model, input_ids, attn, prompt_len):
    with torch.set_grad_enabled(model.training):
        out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    tgt = input_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    g = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = attn[:, 1:].float()
    if prompt_len - 1 > 0:
        mask[:, :prompt_len - 1] = 0.0
    return (g * mask).sum(dim=-1)


def _grpo_step(policy, ref, tokenizer, prompt, gold, lang) -> tuple[torch.Tensor, dict]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=MAX_SEQ_LEN - GEN_MAX_NEW).to(policy.device)
    prompt_len = enc["input_ids"].shape[1]

    policy.eval()
    with torch.no_grad():
        gen = policy.generate(
            **enc, do_sample=True, temperature=GEN_TEMP, top_p=0.95,
            num_return_sequences=G_SAMPLES, max_new_tokens=GEN_MAX_NEW,
            pad_token_id=tokenizer.eos_token_id,
        )
    policy.train()

    rewards, fmts, contents, hallus = [], [], [], []
    for i in range(gen.shape[0]):
        txt = tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=True)
        r, parts = sasr_reward.composite_reward(txt, gold, prompt)
        rewards.append(r); fmts.append(parts["r_format"])
        contents.append(parts["r_content"]); hallus.append(parts["r_hallu"])
    rewards_t = torch.tensor(rewards, device=policy.device, dtype=torch.float32)
    adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    attn = (gen != pad).long()
    attn[:, :prompt_len] = 1

    logp_pi = _logprobs_completion(policy, gen, attn, prompt_len)
    with torch.no_grad():
        logp_ref = _logprobs_completion(ref, gen, attn, prompt_len)
    logp_old = logp_pi.detach()

    log_ratio = logp_pi - logp_old
    ratio = torch.exp(log_ratio)
    surrogate = -torch.min(ratio * adv,
                            torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv).mean()
    lrf = logp_pi - logp_ref
    kl = (torch.exp(-lrf) - 1.0 + lrf).mean()
    loss = surrogate + KL_BETA * kl

    return loss, {
        "reward_mean": float(rewards_t.mean()), "reward_std": float(rewards_t.std()),
        "format_mean": float(np.mean(fmts)), "content_mean": float(np.mean(contents)),
        "hallu_mean": float(np.mean(hallus)), "kl": float(kl), "lang": lang,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="Stage A LoRA adapter path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kg", default="/workspace/scripts/dorosz_ckg.json",
                    help="Dorosz CKG json for anti-hallu vocab")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    log.info("Loading SASR subset (n=%d)...", args.n)
    ds = load_medunified_sasr_subset(n=args.n)
    log.info("  loaded %d rows | langs: %s", len(ds), {l: sum(1 for r in ds if r["lang"] == l) for l in ("en","fr","darija")})

    log.info("Configuring SASR reward (SB_CS + anti-hallu)...")
    drug_vocab = _load_kg_drug_vocab(args.kg)
    sasr_reward.configure(drug_vocab=drug_vocab,
                          sent_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                          w_format=0.20, w_content=0.80, w_hallu=0.30)
    log.info("  vocab size: %d", len(drug_vocab))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    log.info("Loading base + adapter (policy trainable, ref frozen)...")
    base_p = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
        device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    base_p.config.use_cache = False
    base_p = prepare_model_for_kbit_training(base_p, use_gradient_checkpointing=True)
    policy = PeftModel.from_pretrained(base_p, args.adapter, is_trainable=True)

    base_r = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
        device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    ref = PeftModel.from_pretrained(base_r, args.adapter)
    ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=args.cache_dir)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    optim = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                lr=LR, weight_decay=0.0)
    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project="helix-v2", name=f"stage-b-sasr-{int(time.time())}")

    ds = ds.shuffle(seed=SEED)
    it = iter(ds)
    def _next(n):
        out = []
        for _ in range(n):
            try: out.append(next(it))
            except StopIteration:
                nonlocal_iter = iter(ds.shuffle(seed=SEED + 1)); out.append(next(nonlocal_iter))
        return out

    # ── Warmup SFT ────────────────────────────────────────────────────────
    log.info("Warmup %d SFT steps on the SASR subset...", WARMUP_STEPS)
    policy.train()
    step, micro = 0, 0
    while step < WARMUP_STEPS:
        rows = _next(PER_DEVICE_BATCH)
        from data_v2 import _fmt_sft
        loss = _sft_step(policy, tok, [_fmt_sft({**r, "language": r["lang"]}) for r in rows])
        (loss / GRAD_ACCUM).backward()
        micro += 1
        if micro == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step(); optim.zero_grad(); micro = 0; step += 1
            if step % 10 == 0:
                log.info("  warmup step %d/%d  sft_loss=%.4f", step, WARMUP_STEPS, float(loss))
                if use_wandb: wandb.log({"warmup/sft_loss": float(loss), "step": step})

    # ── Adaptive SASR ─────────────────────────────────────────────────────
    log.info("=== Adaptive SASR phase (%d steps, G=%d) ===", SASR_STEPS, G_SAMPLES)
    p_t = 0.5  # initial SFT/GRPO mix
    for s in range(SASR_STEPS):
        rows = _next(1)
        r = rows[0]
        # Roll a coin weighted by p_t (SFT prob) for each accumulation slot
        do_sft = (np.random.rand() < p_t)
        if do_sft:
            from data_v2 import _fmt_sft
            loss = _sft_step(policy, tok, [_fmt_sft({**r, "language": r["lang"]})])
            stats = {"phase": "sft", "loss": float(loss)}
        else:
            loss, stats = _grpo_step(policy, ref, tok, r["prompt"], r["gold"], r["lang"])
            stats["phase"] = "grpo"; stats["loss"] = float(loss)
        loss.backward()
        if (s + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step(); optim.zero_grad()
            # Adapt p_t every accum: based on reward trend (simple proxy)
            if "reward_mean" in stats and stats["reward_mean"] > 0.5:
                p_t = max(0.05, p_t - 0.005)  # more GRPO when reward is healthy
            elif "reward_mean" in stats and stats["reward_mean"] < 0.2:
                p_t = min(0.7, p_t + 0.01)    # back off to SFT when reward weak
        if (s + 1) % 25 == 0:
            log.info("  sasr %d/%d  phase=%s loss=%.3f p_t=%.2f reward=%.3f",
                     s+1, SASR_STEPS, stats["phase"], stats["loss"], p_t, stats.get("reward_mean", 0))
            if use_wandb:
                wandb.log({**{f"sasr/{k}": v for k, v in stats.items() if isinstance(v, (int,float))},
                            "sasr/p_t": p_t, "step": s+1})

    log.info("Saving SASR LoRA -> %s", args.out)
    policy.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    log.info("Stage B DONE.")


if __name__ == "__main__":
    main()

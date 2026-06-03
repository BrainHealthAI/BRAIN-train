"""
Log-likelihood MCQ scoring for HELIX-v2 — the standard `lm-eval-harness` methodology.

For each MCQ item, we build the prompt "Q ... options ... The answer is:" and read the
model's logits at the FINAL position. We then compare logP(' A'), logP(' B'),
logP(' C'), logP(' D') (or just the first 3 if PubMedQA) and pick argmax.

Why this matters for our CoT-trained model:
  - No generation → no truncation by `max_new_tokens`
  - No parser → no failure from prose-only answers (esp. Darija)
  - Single forward pass per item → ~10-15× faster than generation-based eval
  - Comparable to literature: Meditron, BioMistral, Open LLM Leaderboard all use this

Run on the pod (after the generation-based eval finishes):
    python reeval_loglikelihood.py \\
        --base    /workspace/models/brainmed-base-v1 \\
        --adapter /workspace/models/brainmed-cot-v2 \\
        --out     /workspace/outputs/eval_v2_loglikelihood.json \\
        --mcq-limit 200
"""
from __future__ import annotations

import argparse, json, logging, os, time
from typing import Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reeval_ll")


def build_prompt(question: str, options: Sequence[str]) -> str:
    """Generic 0-shot MCQ prompt — standard lm-eval-harness style."""
    letters = "ABCD"[: len(options)]
    body = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a medical exam assistant. Answer with the single best option.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n\n{body}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:"
    )


def precompute_letter_ids(tok) -> dict[str, int]:
    """For each letter A-D, find the token id of ' A', ' B', ' C', ' D' (the leading space matters)."""
    ids = {}
    for letter in "ABCD":
        candidate = " " + letter
        toks = tok.encode(candidate, add_special_tokens=False)
        if len(toks) == 1:
            ids[letter] = toks[0]
        else:
            # Some tokenizers split " A" into 2 tokens; fall back to the LAST token
            # (which is the letter itself, typically). This is robust for Llama-3.
            ids[letter] = toks[-1]
    return ids


def load_model(base: str, adapter: str, cache: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        cache_dir=cache, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    m.config.use_cache = True
    m = PeftModel.from_pretrained(m, adapter, cache_dir=cache)
    m.eval()
    tok = AutoTokenizer.from_pretrained(base, cache_dir=cache)
    tok.pad_token = tok.eos_token
    return m, tok


@torch.no_grad()
def score_item(model, tok, letter_ids, question, options) -> str:
    """Returns the predicted letter (A/B/C/D)."""
    prompt = build_prompt(question, options)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4000).to(model.device)
    out = model(**enc, use_cache=False)
    last_logits = out.logits[0, -1, :].float()  # (vocab,)
    n = len(options)
    scores = {letter: last_logits[letter_ids[letter]].item() for letter in "ABCD"[:n]}
    return max(scores, key=scores.get)


def eval_mcq(model, tok, letter_ids, items, name, limit):
    items = items[:limit]
    n_ok = 0
    t0 = time.time()
    for i, it in enumerate(items):
        pred = score_item(model, tok, letter_ids, it["question"], it["choices"])
        if pred == it["correct_letter"].upper():
            n_ok += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info("  %s %d/%d  acc=%.3f  (%.2fs/item)",
                     name, i + 1, len(items), n_ok / (i + 1), elapsed / (i + 1))
    return {"acc": n_ok / max(1, len(items)), "n": len(items), "name": name}


def load_benchmarks(token: str | None):
    from datasets import load_dataset
    out = {}
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        out["MedQA-USMLE"] = [{"question": r["question"],
                                "choices": [r["options"][k] for k in ["A","B","C","D"]],
                                "correct_letter": r["answer_idx"]} for r in ds]
    except Exception as e: log.warning("MedQA-USMLE skip: %s", e)
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        out["MedMCQA"] = [{"question": r["question"],
                            "choices": [r["opa"], r["opb"], r["opc"], r["opd"]],
                            "correct_letter": "ABCD"[int(r["cop"])]} for r in ds if r.get("question")]
    except Exception as e: log.warning("MedMCQA skip: %s", e)
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        m = {"yes": "A", "no": "B", "maybe": "C"}
        out["PubMedQA"] = [{"question": r["question"],
                             "choices": ["yes", "no", "maybe"],
                             "correct_letter": m[r["final_decision"]]} for r in ds]
    except Exception as e: log.warning("PubMedQA skip: %s", e)
    for sub in ("clinical_knowledge", "medical_genetics", "professional_medicine", "college_medicine"):
        try:
            ds = load_dataset("cais/mmlu", sub, split="test")
            out[f"MMLU-{sub}"] = [{"question": r["question"], "choices": r["choices"],
                                    "correct_letter": "ABCD"[int(r["answer"])]} for r in ds]
        except Exception as e: log.warning("MMLU-%s skip: %s", sub, e)
    try:
        ds = load_dataset("BrainHealthAI/MedQA-Darija-MCQ-500", split="test", token=token)
        out["Darija-MCQ-500"] = [{"question": r["question"], "choices": r["choices"],
                                    "correct_letter": r["correct_letter"]} for r in ds]
    except Exception as e: log.warning("Darija MCQ skip: %s", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcq-limit", type=int, default=200)
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    tok_hf = os.environ.get("HF_TOKEN")
    log.info("Loading model %s + adapter %s", args.base, args.adapter)
    model, tokenizer = load_model(args.base, args.adapter, args.cache_dir)
    letter_ids = precompute_letter_ids(tokenizer)
    log.info("letter token ids: %s", letter_ids)

    log.info("Loading benchmarks...")
    benches = load_benchmarks(tok_hf)
    log.info("benchmarks loaded: %s", list(benches.keys()))

    results = {
        "adapter": args.adapter,
        "method": "log_likelihood_argmax",
        "prompt_style": "options + 'The answer is:'",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mcq": {},
    }
    for name, items in benches.items():
        log.info("MCQ: %s  (n=%d, limit=%d)", name, len(items), args.mcq_limit)
        results["mcq"][name] = eval_mcq(model, tokenizer, letter_ids, items, name, args.mcq_limit)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("saved %s", args.out)
    print("\n=== LOG-LIKELIHOOD EVAL SUMMARY ===")
    for name, r in results["mcq"].items():
        print(f"  MCQ {name:32}  acc={r['acc']:.3f}  n={r['n']}")


if __name__ == "__main__":
    main()

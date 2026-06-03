"""
Final eval — combines log-likelihood MCQ (fast, lit-comparable) + generation per-language.

Designed to be run AFTER killing the slow generation-based MCQ eval. It produces
ONE consolidated eval_v2_final.json that contains both:
  - mcq:           accuracy per benchmark via log-likelihood argmax (no truncation, no parser)
  - generation:    SB_CS + ROUGE-L per language, on MedUnified held-out (think-stripped)

~70 min total on L40S for limit=200 MCQ + limit=100 gen.

Usage:
    python final_eval.py \\
        --base    /workspace/models/brainmed-base-v1 \\
        --adapter /workspace/models/brainmed-cot-v2 \\
        --out     /workspace/outputs/eval_v2_final.json \\
        --mcq-limit 200 --gen-limit 100
"""
from __future__ import annotations

import argparse, json, logging, os, re, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("final_eval")

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_think(t): return THINK_RE.sub("", t or "").strip()


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(base: str, adapter: str, cache: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
        device_map="auto", cache_dir=cache, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    m.config.use_cache = True
    m = PeftModel.from_pretrained(m, adapter, cache_dir=cache); m.eval()
    tok = AutoTokenizer.from_pretrained(base, cache_dir=cache)
    tok.pad_token = tok.eos_token
    return m, tok


# ── Log-likelihood MCQ scoring ─────────────────────────────────────────────

def mcq_prompt(question: str, options: list[str]) -> str:
    letters = "ABCD"[: len(options)]
    body = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a medical exam assistant. Answer with the single best option.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}\n\n{body}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\nThe answer is:")


def precompute_letter_ids(tok):
    ids = {}
    for L in "ABCD":
        toks = tok.encode(" " + L, add_special_tokens=False)
        ids[L] = toks[-1]
    return ids


@torch.no_grad()
def score_mcq_item(model, tok, letter_ids, question, options):
    prompt = mcq_prompt(question, options)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4000).to(model.device)
    out = model(**enc, use_cache=False)
    last_logits = out.logits[0, -1, :].float()
    n = len(options)
    scores = {L: last_logits[letter_ids[L]].item() for L in "ABCD"[:n]}
    return max(scores, key=scores.get)


def eval_mcq(model, tok, letter_ids, items, name, limit):
    items = items[:limit]; n_ok = 0; t0 = time.time()
    for i, it in enumerate(items):
        pred = score_mcq_item(model, tok, letter_ids, it["question"], it["choices"])
        if pred == it["correct_letter"].upper(): n_ok += 1
        if (i + 1) % 50 == 0:
            log.info("  %s %d/%d  acc=%.3f  (%.2fs/it)",
                     name, i + 1, len(items), n_ok/(i+1), (time.time()-t0)/(i+1))
    return {"acc": n_ok / max(1, len(items)), "n": len(items), "name": name}


# ── Generation eval ────────────────────────────────────────────────────────

def gen_prompt(question: str) -> str:
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a medical reasoning assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


@torch.no_grad()
def gen_one(model, tok, prompt, max_new=512):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_gen(model, tok, items, limit):
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    items = items[:limit]; rs, cs, ls = [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        out = strip_think(gen_one(model, tok, gen_prompt(it["question"]), max_new=512))
        gold = it["answer"]
        rs.append(rouge.score(gold, out)["rougeL"].fmeasure)
        embs = sbert.encode([out, gold], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
        cs.append(max(0.0, float(np.dot(embs[0], embs[1]))))
        ls.append(len(out))
        if (i + 1) % 20 == 0:
            log.info("  gen %d/%d  rouge=%.3f sb_cs=%.3f  (%.1fs/it)",
                     i + 1, len(items), np.mean(rs), np.mean(cs), (time.time()-t0)/(i+1))
    return {"rouge_l": float(np.mean(rs)), "sb_cs": float(np.mean(cs)),
            "n": len(items), "mean_out_len": float(np.mean(ls))}


# ── Benchmarks ─────────────────────────────────────────────────────────────

def load_benchmarks(tok_hf):
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
        out["PubMedQA"] = [{"question": r["question"], "choices": ["yes","no","maybe"],
            "correct_letter": m[r["final_decision"]]} for r in ds]
    except Exception as e: log.warning("PubMedQA skip: %s", e)
    for sub in ("clinical_knowledge","medical_genetics","professional_medicine","college_medicine"):
        try:
            ds = load_dataset("cais/mmlu", sub, split="test")
            out[f"MMLU-{sub}"] = [{"question": r["question"], "choices": r["choices"],
                "correct_letter": "ABCD"[int(r["answer"])]} for r in ds]
        except Exception as e: log.warning("MMLU-%s skip: %s", sub, e)
    try:
        ds = load_dataset("BrainHealthAI/MedQA-Darija-MCQ-500", split="test", token=tok_hf)
        out["Darija-MCQ-500"] = [{"question": r["question"], "choices": r["choices"],
            "correct_letter": r["correct_letter"]} for r in ds]
    except Exception as e: log.warning("Darija MCQ skip: %s", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcq-limit", type=int, default=200)
    ap.add_argument("--gen-limit", type=int, default=100)
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    tok_hf = os.environ.get("HF_TOKEN")
    log.info("Loading model %s + adapter %s", args.base, args.adapter)
    model, tokenizer = load_model(args.base, args.adapter, args.cache_dir)
    letter_ids = precompute_letter_ids(tokenizer)
    log.info("letter token ids: %s", letter_ids)

    res = {"adapter": args.adapter, "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mcq": {}, "mcq_method": "log_likelihood_argmax",
            "generation": {"per_language": {}}, "generation_method": "free_form_then_strip_think"}

    # PART 1 — log-likelihood MCQ (fast, ~15-20 min total)
    log.info("=== PART 1: log-likelihood MCQ ===")
    benches = load_benchmarks(tok_hf)
    log.info("benchmarks: %s", list(benches.keys()))
    for name, items in benches.items():
        log.info("MCQ: %s  (n=%d, limit=%d)", name, len(items), args.mcq_limit)
        res["mcq"][name] = eval_mcq(model, tokenizer, letter_ids, items, name, args.mcq_limit)
    # Save intermediate (in case gen eval crashes, MCQ is preserved)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log.info("MCQ done, intermediate saved to %s", args.out)

    # PART 2 — generation eval (slower, ~50 min)
    log.info("=== PART 2: generation eval per language ===")
    try:
        ds = load_dataset("BrainHealthAI/MedUnified",
            data_files={"validation": "eval.parquet"}, token=tok_hf)["validation"]
        intern = [{"question": r["question"], "answer": r["answer"], "language": r["language"]}
                   for r in ds]
        per = {}
        for lang in ("en", "fr", "darija"):
            sub = [r for r in intern if r["language"] == lang]
            if sub:
                log.info("Gen [%s]: n=%d limit=%d", lang, len(sub), args.gen_limit)
                per[lang] = eval_gen(model, tokenizer, sub, args.gen_limit)
        res["generation"]["per_language"] = per
    except Exception as e:
        log.warning("Gen eval failed: %s", e)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log.info("FINAL eval saved to %s", args.out)

    print("\n=== FINAL EVAL SUMMARY ===")
    print(f"  Method: MCQ via log-likelihood argmax, Gen via free-form + strip-think")
    for name, r in res["mcq"].items():
        print(f"  MCQ {name:32}  acc={r['acc']:.3f}  n={r['n']}")
    for lang, r in res["generation"]["per_language"].items():
        print(f"  Gen [{lang:6}]  rouge_l={r['rouge_l']:.3f}  sb_cs={r['sb_cs']:.3f}  n={r['n']}")


if __name__ == "__main__":
    main()

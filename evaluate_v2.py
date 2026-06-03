"""
Evaluate a HELIX-FT v2 adapter on internal + external medical benchmarks.

Internal:
  - MedUnified eval (495 held-out, single-language) — generation metrics

External MCQ benchmarks (English unless noted):
  - GBaker/MedQA-USMLE-4-options (USMLE)
  - openlifescienceai/medmcqa (validation)
  - qiaojin/PubMedQA (pqa_labeled)
  - cais/mmlu: clinical_knowledge / medical_genetics / professional_medicine / college_medicine
  - BrainHealthAI/MedQA-Darija-MCQ-500 (our own, Darija) — if pushed

For each MCQ: ask the model "answer with the letter A/B/C/D only", parse, compute accuracy.
For generation: BLEU, ROUGE-L, METEOR, BERTScore-F1, SB_CS.

Run on the pod (after Stage C):
  python evaluate_v2.py --base /workspace/models/brainmed-base-v1 \\
                        --adapter /workspace/models/brainmed-cot-v2 \\
                        --out /workspace/outputs/eval_v2.json \\
                        [--skip-baseline] [--skip-sft-big] [--limit 200]
"""
from __future__ import annotations

import argparse, json, logging, os, re, sys, time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_v2")

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


# ── Generators ─────────────────────────────────────────────────────────────

def load_model(base: str, adapter: str | None, cache: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
        device_map="auto", cache_dir=cache, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    m.config.use_cache = True
    if adapter:
        m = PeftModel.from_pretrained(m, adapter, cache_dir=cache)
    m.eval()
    tok = AutoTokenizer.from_pretrained(base, cache_dir=cache)
    tok.pad_token = tok.eos_token
    return m, tok


def generate(model, tok, prompt: str, max_new: int = 128, temperature: float = 0.0):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=temperature > 0,
                              temperature=max(0.001, temperature),
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ── MCQ scorers ────────────────────────────────────────────────────────────

def mcq_prompt(question: str, options: list[str]) -> str:
    letters = "ABCD"[:len(options)]
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a medical exam assistant. Choose the single best option. Reply with ONLY the letter (A, B, C, or D), no explanation.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n\n{opts}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_letter(text: str) -> str | None:
    if not text: return None
    m = LETTER_RE.search(text.strip())
    return m.group(1).upper() if m else None


def eval_mcq(model, tok, items: list[dict], desc: str, limit: int = 200) -> dict:
    items = items[:limit]
    n_ok, n_total = 0, 0
    for i, it in enumerate(items):
        out = generate(model, tok, mcq_prompt(it["question"], it["choices"]), max_new=8, temperature=0.0)
        pred = parse_letter(out)
        gold = it["correct_letter"].upper()
        if pred == gold: n_ok += 1
        n_total += 1
        if (i+1) % 25 == 0:
            log.info("  %s %d/%d  acc=%.3f", desc, i+1, len(items), n_ok / max(1, n_total))
    return {"acc": n_ok / max(1, n_total), "n": n_total, "name": desc}


# ── Generation scorer ──────────────────────────────────────────────────────

def eval_generation(model, tok, items: list[dict], limit: int = 100) -> dict:
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    items = items[:limit]
    rouges, sbcs, lens_in, lens_out = [], [], [], []
    for i, it in enumerate(items):
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a medical reasoning assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{it['question']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        out = generate(model, tok, prompt, max_new=512, temperature=0.0)
        gold = it["answer"]
        rouges.append(rouge.score(gold, out)["rougeL"].fmeasure)
        embs = sbert.encode([out, gold], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sbcs.append(max(0.0, float(np.dot(embs[0], embs[1]))))
        lens_out.append(len(out))
        if (i+1) % 20 == 0:
            log.info("  gen %d/%d  rouge=%.3f sbcs=%.3f", i+1, len(items),
                     np.mean(rouges), np.mean(sbcs))
    return {"rouge_l": float(np.mean(rouges)), "sb_cs": float(np.mean(sbcs)),
            "n": len(items), "mean_out_len": float(np.mean(lens_out))}


# ── Benchmark loaders ──────────────────────────────────────────────────────

def load_mcq_benchmarks(token: str | None = None) -> dict[str, list[dict]]:
    from datasets import load_dataset
    out = {}
    def _norm(items, qf, optsf, ansf):
        return [{"question": r[qf], "choices": r[optsf] if isinstance(r[optsf], list)
                  else [r[f] for f in optsf],
                  "correct_letter": (r[ansf] if isinstance(r[ansf], str) else "ABCD"[int(r[ansf])])}
                 for r in items if r.get(qf)]
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        out["MedQA-USMLE"] = [{"question": r["question"],
                                "choices": [r["options"][k] for k in ["A","B","C","D"]],
                                "correct_letter": r["answer_idx"]} for r in ds]
    except Exception as e: log.warning("MedQA-USMLE skip: %s", e)
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        idx_to = ["A","B","C","D"]
        out["MedMCQA"] = [{"question": r["question"],
                            "choices": [r["opa"], r["opb"], r["opc"], r["opd"]],
                            "correct_letter": idx_to[int(r["cop"])]} for r in ds if r.get("question")]
    except Exception as e: log.warning("MedMCQA skip: %s", e)
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        # PubMedQA is yes/no/maybe — map to A/B/C
        m = {"yes": "A", "no": "B", "maybe": "C"}
        out["PubMedQA"] = [{"question": r["question"],
                             "choices": ["yes", "no", "maybe"],
                             "correct_letter": m[r["final_decision"]]} for r in ds]
    except Exception as e: log.warning("PubMedQA skip: %s", e)
    for sub in ("clinical_knowledge","medical_genetics","professional_medicine","college_medicine"):
        try:
            ds = load_dataset("cais/mmlu", sub, split="test")
            out[f"MMLU-{sub}"] = [{"question": r["question"], "choices": r["choices"],
                                    "correct_letter": "ABCD"[int(r["answer"])]} for r in ds]
        except Exception as e: log.warning("MMLU-%s skip: %s", sub, e)
    try:
        ds = load_dataset("BrainHealthAI/MedQA-Darija-MCQ-500", split="test", token=token)
        out["Darija-MCQ-500"] = [{"question": r["question"], "choices": r["choices"],
                                    "correct_letter": r["correct_letter"]} for r in ds]
    except Exception as e: log.warning("Darija MCQ skip (not pushed yet?): %s", e)
    return out


def load_internal_eval(token: str | None = None) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("BrainHealthAI/MedUnified",
                          data_files={"validation": "eval.parquet"}, token=token)["validation"]
        return [{"question": r["question"], "answer": r["answer"],
                  "language": r["language"], "source": r["source"]} for r in ds]
    except Exception as e:
        log.warning("MedUnified eval skip: %s", e)
        return []


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcq-limit", type=int, default=200)
    ap.add_argument("--gen-limit", type=int, default=100)
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME") or "/workspace/cache")
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    log.info("Loading model: %s + adapter %s", args.base, args.adapter)
    model, tokenizer = load_model(args.base, args.adapter, args.cache_dir)

    results = {"adapter": args.adapter, "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "mcq": {}, "generation": {}}

    log.info("Loading MCQ benchmarks...")
    benches = load_mcq_benchmarks(token=tok)
    for name, items in benches.items():
        log.info("MCQ: %s (%d items, limit=%d)", name, len(items), args.mcq_limit)
        results["mcq"][name] = eval_mcq(model, tokenizer, items, name, limit=args.mcq_limit)

    log.info("Loading internal generation eval...")
    intern = load_internal_eval(token=tok)
    by_lang = {}
    for lang in ("en", "fr", "darija"):
        sub = [r for r in intern if r["language"] == lang]
        if not sub: continue
        log.info("Gen eval [%s]: %d items (limit=%d)", lang, len(sub), args.gen_limit)
        by_lang[lang] = eval_generation(model, tokenizer, sub, limit=args.gen_limit)
    results["generation"]["per_language"] = by_lang

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved %s", args.out)

    # Compact summary
    print("\n=== EVAL SUMMARY ===")
    for name, r in results["mcq"].items():
        print(f"  MCQ {name:32}  acc={r['acc']:.3f}  n={r['n']}")
    for lang, r in by_lang.items():
        print(f"  Gen [{lang}]                       rouge_l={r['rouge_l']:.3f}  sb_cs={r['sb_cs']:.3f}  n={r['n']}")


if __name__ == "__main__":
    main()

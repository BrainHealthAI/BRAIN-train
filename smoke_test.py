"""smoke_test.py — Pre-flight check for the BRAIN HEALTH training pipeline.

Validates every module that does NOT require loading the 8B model.
Designed to run in a few minutes on any machine (laptop, RTX 3050, CPU-only…).
On RunPod, also useful as a sanity check before the real launch.

Tests:
  T1  Imports + Python deps
  T2  Dorosz KG loader (loads, drug NER on a sample)
  T3  Dataset access — Big (MedQA-Darija-MultiLingual, 5 rows trilingual check)
  T4  Dataset access — Small (BrainHealthAI/MedQA_mutilangual, 5 rows)
  T5  Tokenizer + chat template + prompt building (Llama-3.1-8B-Instruct, ~1MB tokenizer only)
  T6  Reward primitives (format / content / composite) on synthetic predictions
  T7  Evaluation metrics (BLEU/GLEU/ROUGE/METEOR/Token-PRF/BERTScore[opt]/SB_CS)
  T8  Plot generation with dummy comparison.json
  T9  Configuration sanity (paths writable, GPU available, etc.)

Run:
  python smoke_test.py
  python smoke_test.py --skip-bertscore       # if torch out-of-memory on small GPU
  python smoke_test.py --hf-cache /tmp/hf     # custom HF cache dir
"""
from __future__ import annotations
import argparse, json, os, sys, traceback, tempfile, time
from pathlib import Path

# Make config + sibling modules importable when run from the brain_health_training/ dir.
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))


import sys as _sys
# Force UTF-8 on stdout/stderr (Windows defaults to cp1252 which breaks emoji)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; END = "\033[0m"

def ok(msg):    print(f"  {GREEN}[OK]{END}   {msg}")
def fail(msg):  print(f"  {RED}[FAIL]{END} {msg}")
def warn(msg):  print(f"  {YELLOW}[WARN]{END} {msg}")
def step(msg):  print(f"\n{CYAN}{msg}{END}")


# ── T1 imports ────────────────────────────────────────────────────────────
def t1_imports():
    step("T1  Python imports")
    import importlib
    deps = [
        ("torch",              "≥2.3"),
        ("transformers",       "≥4.45"),
        ("datasets",           "≥2.20"),
        ("peft",               "≥0.12"),
        ("trl",                "≥0.11"),
        ("rouge_score",        ""),
        ("nltk",               ""),
        ("sentence_transformers", ""),
        ("matplotlib",         ""),
    ]
    optional = [("bert_score", ""), ("bitsandbytes", "GPU only")]
    for name, hint in deps:
        try:
            m = importlib.import_module(name)
            v = getattr(m, "__version__", "")
            ok(f"{name:<22} {v}  {hint}")
        except Exception as e:
            fail(f"{name}: {e}")
            return False
    for name, hint in optional:
        try:
            m = importlib.import_module(name)
            ok(f"{name:<22} {getattr(m,'__version__','')}  {hint}")
        except Exception:
            warn(f"{name}: not installed (optional, used by RunPod stage)")

    # NLTK data
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
        ok("nltk.punkt corpus available")
    except LookupError:
        warn("nltk.punkt missing — downloading…")
        import nltk
        nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
        ok("nltk corpora downloaded")

    return True


# ── T2 Dorosz KG ──────────────────────────────────────────────────────────
def t2_dorosz_kg():
    step("T2  Dorosz Causal KG loader")
    from dorosz_kg import DoroszKG, get_kg
    kg_path = HERE / "dorosz_ckg.json"
    if not kg_path.exists():
        fail(f"KG file missing: {kg_path}")
        return False
    kg = DoroszKG(kg_path)
    if kg.empty:
        fail("KG loaded as empty")
        return False
    ok(f"KG loaded: {len(kg.name_to_id)} drug names")

    # NER on synthetic medical text in 3 languages
    samples = [
        "Le patient prend de la metformine 500mg deux fois par jour pour son diabète.",
        "Patient receives amoxicilline 1g tid for the bacterial infection.",
        "المريض كياخد paracétamol 1g كل 6 ساعات للحمى",
    ]
    n_total = 0
    for s in samples:
        facts = kg.lookup_facts(s, max_drugs=3)
        n_total += len(facts)
        ok(f"  scan «{s[:40]}…» → {len(facts)} drugs")
        for f in facts:
            print(f"        {f}")
    if n_total == 0:
        warn("No drug found across 3 samples — KG drug index might be too noisy "
             "(this is acceptable; KG augmentation will be sparse).")
    return True


# ── T3 Big dataset ────────────────────────────────────────────────────────
def t3_dataset_big(args):
    """Verify schema + grab a few non-audio rows. Avoids torchcodec dependency
    by reading the underlying parquet directly with pyarrow."""
    step("T3  Big dataset access (Williamsanderson/MedQA-Darija-MultiLingual)")

    # ── Schema check via HF dataset builder (fast, no data download) ──────
    from datasets import load_dataset_builder
    try:
        b = load_dataset_builder("Williamsanderson/MedQA-Darija-MultiLingual",
                                  "default", cache_dir=args.hf_cache)
        feats = list(b.info.features.keys())
    except Exception as e:
        fail(f"load_dataset_builder failed: {e}")
        return False
    expected = ["question_en", "answer_en", "question_fr", "answer_fr",
                "question_darija", "answer_darija", "specialty_id", "source"]
    for col in expected:
        if col not in feats:
            fail(f"missing column: {col} (have: {feats})")
            return False
    audio_cols = [c for c in feats if c.startswith("audio_")]
    ok(f"schema OK ({len(feats)} cols, {len(audio_cols)} audio drop-able, "
       f"trilingual question_*/answer_* present)")

    # ── Pull 5 actual rows via pyarrow (skips audio decode entirely) ──────
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        # The default config is in `data/train.parquet` shards. Take the first.
        path = hf_hub_download(
            "Williamsanderson/MedQA-Darija-MultiLingual",
            "data/train.parquet", repo_type="dataset", cache_dir=args.hf_cache,
            token=os.environ.get("HF_TOKEN"))
        pf = pq.ParquetFile(path)
        # read only the text columns we care about (NOT audio)
        text_cols = [c for c in pf.schema.names if not c.startswith("audio_")]
        first_batch = next(pf.iter_batches(batch_size=5, columns=text_cols))
        rows = first_batch.to_pylist()
    except Exception as e:
        warn(f"parquet read fallback failed: {e}; schema-only check passed.")
        return True
    ok(f"read 5 rows directly from parquet ({len(rows[0])} non-audio columns)")
    sample = rows[0]
    print(f"      pair_id  : {sample.get('pair_id', '<none>')}")
    print(f"      specialty: {sample.get('specialty_id')}")
    print(f"      Q[en]    : {(sample.get('question_en') or '')[:80]}…")
    print(f"      A[fr]    : {(sample.get('answer_fr') or '')[:80]}…")
    print(f"      Q[darija]: {(sample.get('question_darija') or '')[:80]}…")
    return True


# ── T4 Small dataset ──────────────────────────────────────────────────────
def t4_dataset_small(args):
    step("T4  Small dataset access (BrainHealthAI/MedQA_mutilangual)")
    from datasets import load_dataset
    try:
        ds = load_dataset("BrainHealthAI/MedQA_mutilangual", "default",
                          split="train", cache_dir=args.hf_cache, streaming=True)
        rows = list(ds.take(5))
    except Exception as e:
        fail(f"load_dataset failed: {e}")
        return False
    ok(f"streamed 5 rows")
    expected = ["question", "context_question", "answer", "language", "speciality"]
    for col in expected:
        if col not in rows[0]:
            fail(f"missing column: {col}")
            return False
    ok("schema OK (question + context_question + answer + language + speciality)")
    sample = rows[0]
    print(f"      lang     : {sample.get('language')}")
    print(f"      ctx      : {(sample.get('context_question') or '')[:80]}…")
    print(f"      Q        : {(sample.get('question') or '')[:80]}…")
    print(f"      A        : {(sample.get('answer') or '')[:80]}…")
    return True


# ── T5 Tokenizer + chat template + prompt building ───────────────────────
def t5_tokenizer_and_prompt(args):
    step("T5  Tokenizer + chat template + prompt building (Llama-3.1-8B)")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=args.hf_cache, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            warn("Llama-3.1 is gated — set HF_TOKEN env var with a token that "
                 "has accepted the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            warn("This test is SKIPPED locally; the runpod setup will need this token.")
            return True  # not a blocker for the rest
        fail(f"tokenizer download failed: {e}")
        return False
    ok(f"tokenizer downloaded (vocab={len(tok)})")

    # Test data.py prompt building for both datasets
    import data as D
    fake_big_row = {
        "question_en":      "What are common symptoms of type 2 diabetes?",
        "answer_en":        "Increased thirst, frequent urination, fatigue, and blurred vision.",
        "question_fr":      "Quels sont les symptômes courants du diabète de type 2 ?",
        "answer_fr":        "Soif accrue, urines fréquentes, fatigue et vision floue.",
        "question_darija":  "شنو هما الأعراض المعروفة ديال السكري من النوع الثاني؟",
        "answer_darija":    "العطش الكثير، التبول المتكرر، التعب، والرؤية المشوشة.",
    }
    fake_small_row = {
        "question": "What is the recommended first-line treatment?",
        "context_question": "A 60-year-old man with newly diagnosed type 2 diabetes (HbA1c 8.2%).",
        "answer":   "Metformin 500mg twice daily, with lifestyle modifications.",
        "language": "en",
        "speciality": "endocrinology",
    }

    for lang in ["en", "fr", "darija"]:
        msgs = D._build_messages_big(fake_big_row, lang, with_kg=True)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ok(f"big[{lang}] template OK ({len(text)} chars)")
        for marker in ["<|begin_of_text|>", "system", "user", "assistant",
                       D.C.ANSWER_OPEN, D.C.ANSWER_CLOSE]:
            assert marker in text, f"missing marker '{marker}' in big[{lang}] template"
        # Verify the gold answer is wrapped
        assert fake_big_row[f"answer_{lang}"] in text, f"answer leaked in {lang}"

    msgs_small = D._build_messages_small(fake_small_row)
    text_small = tok.apply_chat_template(msgs_small, tokenize=False, add_generation_prompt=False)
    ok(f"small template OK ({len(text_small)} chars)")
    assert fake_small_row["answer"] in text_small, "small answer not in template"
    assert "Metformin" in text_small

    # Prompt-only (used by SASR rollout + evaluate)
    prompt_msgs = D._build_prompt_only_big(fake_big_row, "fr")
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    ok(f"big prompt-only template OK ({len(prompt_text)} chars)")
    assert fake_big_row["answer_fr"] not in prompt_text, "FR answer leaked into prompt-only !"
    return True


# ── T6 Rewards ────────────────────────────────────────────────────────────
def t6_rewards():
    step("T6  Reward functions (format + content + composite)")
    from train_sasr import format_reward, content_reward, reward_composite, extract_answer

    cases = [
        # (text, expected_format, brief)
        ("<think>Reason here.</think><answer>Final answer.</answer>",          1.0, "both tags"),
        ("<answer>Just answer.</answer>",                                       0.5, "answer only"),
        ("Plain text with no tags.",                                            0.0, "no tags"),
        ("<think>Only think</think>Plain rest.",                                0.0, "think only — no answer"),
    ]
    for text, expected, brief in cases:
        f = format_reward(text)
        if abs(f - expected) > 1e-9:
            fail(f"format_reward({brief!r}) = {f} ; expected {expected}")
            return False
        ok(f"format[{brief}] = {f}")

    # Content reward
    gold = "Metformin 500mg twice daily is first-line therapy for type 2 diabetes."
    pred_good = "<answer>Metformin 500mg twice daily is the standard first-line therapy.</answer>"
    pred_bad  = "<answer>Apple juice three times a week.</answer>"
    pred_complex = "<answer>{symptoms: thirst; diagnostic: T2DM; references: ADA 2024.}</answer>"
    cr_good = content_reward(pred_good, gold, question_tokens=20)
    cr_bad  = content_reward(pred_bad,  gold, question_tokens=20)
    cr_complex = content_reward(pred_complex, gold, question_tokens=40)  # long question, json cues
    if not (cr_good > cr_bad):
        fail(f"content_reward sanity broken: good={cr_good:.3f} ≤ bad={cr_bad:.3f}")
        return False
    ok(f"content[good]={cr_good:.3f}  >  content[bad]={cr_bad:.3f}")
    ok(f"content[complex+jsonkw]={cr_complex:.3f}  (bonus on long Q)")

    # Composite
    r, parts = reward_composite(pred_good, gold, "Some 20-word question " * 1)
    ok(f"composite total={r:.3f}  format={parts['r_format']:.3f}  content={parts['r_content']:.3f}")

    # extract_answer
    assert extract_answer(pred_good) == "Metformin 500mg twice daily is the standard first-line therapy."
    ok("extract_answer parses <answer>…</answer> correctly")
    return True


# ── T7 Evaluation metrics ─────────────────────────────────────────────────
def t7_metrics(args):
    step("T7  Evaluation metrics (per-language)")
    from evaluate import compute_metrics

    # Synthetic preds vs refs (English) — preds are intentionally close to refs
    preds = [
        "Diabetes type 2 is treated with metformin and lifestyle changes.",
        "Hypertension management requires ACE inhibitors and salt restriction.",
        "Acute MI presents with chest pain radiating to the left arm.",
    ]
    refs = [
        "Type 2 diabetes is managed with metformin plus lifestyle modifications.",
        "Hypertension is treated using ACE inhibitors and reduced salt intake.",
        "An acute myocardial infarction causes substernal chest pain radiating to the arm.",
    ]
    if args.skip_bertscore:
        # monkey-patch BERTScore to return zeros (the function is wrapped in try)
        import evaluate as ev
        original = ev.bertscore
        def _stub(preds, refs, **kw):
            import torch
            return torch.zeros(len(preds)), torch.zeros(len(preds)), torch.zeros(len(preds))
        ev.bertscore = _stub
        warn("Skipping BERTScore download (--skip-bertscore).")

    t0 = time.time()
    m = compute_metrics(preds, refs, lang="en")
    print(f"      computed in {time.time()-t0:.1f}s")
    expected_keys = ["BLEU","GLEU","ROUGE_1","ROUGE_2","ROUGE_L","METEOR",
                     "Precision","Recall","F1","BS_P","BS_R","BS_F1","SB_CS","NASS"]
    missing = [k for k in expected_keys if k not in m]
    if missing:
        fail(f"missing metric keys: {missing}")
        return False
    ok(f"all 14 metrics present")
    for k in expected_keys:
        v = m[k]
        if not (0.0 <= v <= 1.0):
            warn(f"  {k} = {v:.3f}  (outside [0,1] — investigate)")
        else:
            print(f"      {k:<10} = {v:.3f}")
    if m["BLEU"] == 0 and m["ROUGE_L"] == 0:
        fail("metrics are all zero — wiring bug")
        return False
    ok("metrics produce sane non-zero values")
    return True


# ── T8 Plots ──────────────────────────────────────────────────────────────
def t8_plots():
    step("T8  Plot generation (with dummy comparison.json)")
    import matplotlib
    matplotlib.use("Agg")
    import config as C, plot_results as P

    # Build a dummy comparison.json structure
    METRICS = P.ALL_METRICS
    MODELS = ["baseline", "sft_small", "sft_big", "sasr_big"]
    import random
    random.seed(0)
    comp = {
        "macro": {m: {k: round(0.3 + 0.6 * random.random(), 3) for k in MODELS}
                  for m in METRICS},
        "per_language": {lang: {m: {k: round(0.3 + 0.6 * random.random(), 3) for k in MODELS}
                                 for m in METRICS} for lang in C.LANGS},
        "deltas_vs_baseline": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp)
        plots_dir = out_root / "plots"; plots_dir.mkdir()
        # Patch config paths to land in tmp
        C.COMPARISON_OUT = out_root / "comparison.json"
        C.PLOTS_DIR      = plots_dir
        C.OUTPUTS_DIR    = out_root
        C.COMPARISON_OUT.write_text(json.dumps(comp))
        P.main()
        produced = sorted(plots_dir.glob("*.png"))
        if len(produced) < 7:
            fail(f"only {len(produced)} plots generated, expected 7")
            return False
        for p in produced:
            print(f"      [OK] {p.name}  ({p.stat().st_size//1024} KB)")
    ok("all 7 plots generated")
    return True


# ── T10 Medical metrics ───────────────────────────────────────────────────
def t10_medical_metrics():
    step("T10 Medical metrics (entity NER, recall, hallucination, P@K, pathology)")
    from medical_metrics import (extract_entities, entity_recall,
                                  hallucination_rate, diagnostic_precision_at_k,
                                  reference_validity, pathology_of, aggregate_medical)

    # Entity extraction
    ref = ("Type 2 diabetes is managed with metformine 500mg twice daily. "
           "Monitor HbA1c every 3 months and screen for complications.")
    pred = ("<answer>Metformine 500mg PO bid is first-line for T2DM. "
            "Order HbA1c test and check blood glucose. May add empagliflozine.</answer>")
    ents = extract_entities(pred)
    if "metformine" not in ents["drugs"]:
        fail(f"NER missed metformine in pred (drugs={ents['drugs']})")
        return False
    ok(f"extract_entities: drugs={len(ents['drugs'])}, symptoms={len(ents['symptoms'])}, "
       f"exams={len(ents['exams'])}, dosages={len(ents['dosages'])}")

    # Entity recall
    rec = entity_recall(pred, ref)
    if rec["recall_drugs"] is None or rec["recall_drugs"] < 0.5:
        warn(f"low drug recall: {rec['recall_drugs']}  (acceptable if NER misses synonyms)")
    ok(f"entity_recall: drugs={rec['recall_drugs']}  exams={rec['recall_exams']}  "
       f"macro={rec['recall_entities_macro']:.3f}")

    # Hallucination
    hal = hallucination_rate(pred, ref)
    ok(f"hallucination: drugs={hal['hallu_drugs']:.3f}  dosage={hal['hallu_dosage']:.3f}  "
       f"macro={hal['hallu_macro']:.3f}")
    if hal["hallu_drugs"] < 0.1:
        warn("hallu_drugs very low — pred and ref share most drugs (expected for matched test)")
    elif hal["hallu_drugs"] > 0.5:
        ok(f"  hallu_drugs {hal['hallu_drugs']:.3f} — captures empagliflozine as 'invented' (not in ref)")

    # Diagnostic Precision@K
    dp = diagnostic_precision_at_k(
        pred="<answer>Likely diagnosis: 1. Type 2 diabetes 2. Metabolic syndrome "
             "3. Pre-diabetes</answer>",
        ref_diagnosis="type 2 diabetes mellitus", k=3)
    if dp["hit_at_k"] != 1:
        fail(f"P@3 broken: {dp}")
        return False
    ok(f"diagnostic_precision_at_k: hit@3=1 with candidates {dp['candidates']}")

    # Reference validity
    rv = reference_validity("See PubMed PMID:12345678 and DOI 10.1234/abc.5678 for details.")
    if rv["n_pmids"] != 1 or rv["n_dois"] != 1:
        fail(f"reference detection broken: {rv}")
        return False
    ok(f"reference_validity: n_dois={rv['n_dois']}, n_pmids={rv['n_pmids']}")

    # Pathology categorization
    cats = [pathology_of("crushing chest pain ST elevation"),
            pathology_of("Le patient a une pneumonie bactérienne"),
            pathology_of("HbA1c is 8.2% with polyuria and polydipsia"),
            pathology_of("Aucune information pertinente.")]
    expected = {"cardiovascular", "infectious", "endocrine"}
    if not expected.issubset(set(cats)):
        fail(f"pathology_of broken — got {cats}, expected to include {expected}")
        return False
    ok(f"pathology_of: {cats}")

    # Aggregate
    rows = [
        {"question_en": "chest pain", "lang": "en"},
        {"question_fr": "j'ai de la fièvre et une infection", "lang": "fr"},
        {"question_en": "metformin dose for diabetes", "lang": "en"},
    ]
    preds = [
        "<answer>Likely MI. ECG and troponin needed.</answer>",
        "<answer>Probable infection bactérienne. Antibiothérapie + paracétamol.</answer>",
        "<answer>Metformine 500mg po bid for T2DM, monitor HbA1c.</answer>",
    ]
    refs = [
        "Acute MI suspected — ECG and serial troponins. Aspirin and clopidogrel.",
        "Infection bactérienne probable — amoxicilline 1g x3/j et paracétamol 1g x4/j.",
        "Metformine 500mg po deux fois par jour. Surveiller HbA1c.",
    ]
    langs = ["en", "fr", "en"]
    agg = aggregate_medical(preds, refs, langs, rows)
    ok(f"aggregate_medical: macro entity_recall={agg['macro']['entity_recall_macro']:.3f}, "
       f"hallu_macro={agg['macro']['hallu_macro']:.3f}, "
       f"per_pathology={list(agg['per_pathology'].keys())}")
    return True


# ── T11 Robustness primitives ─────────────────────────────────────────────
def t11_robustness():
    step("T11 Robustness primitives (paraphrase + ASR-noise)")
    from robustness_tests import paraphrase, simulate_asr_noise

    # Paraphrase
    en_q = "Question: What are the symptoms of type 2 diabetes?"
    paras = paraphrase(en_q, lang="en", n=3, seed=0)
    if len(paras) != 3:
        fail(f"paraphrase returned {len(paras)} items, expected 3")
        return False
    if paras[0] == paras[1] == paras[2] == en_q:
        warn("paraphrase produced no variations (rule didn't apply)")
    else:
        ok(f"paraphrase EN: 3 variants generated, e.g.\n      {paras[0][:90]}")

    fr_q = "Question : Quels sont les symptômes du diabète de type 2 ?"
    paras_fr = paraphrase(fr_q, lang="fr", n=3, seed=0)
    ok(f"paraphrase FR: 3 variants, sample\n      {paras_fr[0][:90]}")

    # ASR noise
    clean = "Patient with chest pain radiating to left arm and dyspnea."
    noisy = simulate_asr_noise(clean, lang="en", seed=1)
    if noisy == clean:
        warn("ASR noise didn't perturb anything (RNG didn't trigger)")
    else:
        ok(f"ASR noise applied:\n      clean : {clean}\n      noisy : {noisy}")
    return True


# ── T9 Config sanity ──────────────────────────────────────────────────────
def t9_config():
    step("T9  Config sanity")
    import config as C, torch
    ok(f"BASE_MODEL              = {C.BASE_MODEL}")
    ok(f"big dataset             = {C.HF_DATASET_BIG}  (limit={C.BIG_LIMIT:,})")
    ok(f"small dataset           = {C.HF_DATASET_SMALL}")
    ok(f"epochs={C.NUM_EPOCHS}  per_device_batch={C.PER_DEVICE_BATCH}  "
       f"effective_batch={C.PER_DEVICE_BATCH*C.GRAD_ACCUM_STEPS}")
    ok(f"max_seq_len={C.MAX_SEQ_LEN}  lr_sft={C.LEARNING_RATE_SFT}  lr_sasr={C.LEARNING_RATE_SASR}")
    ok(f"SASR  warmup={C.SASR_WARMUP_STEPS} γ={C.SASR_GAMMA} G={C.SASR_G_SAMPLES} "
       f"β={C.SASR_BETA} ε={C.SASR_CLIP_EPS}")
    ok(f"reward weights          = format {C.SASR_W_FORMAT} + content {C.SASR_W_CONTENT}")
    ok(f"GPU                     = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"      GPU mem total           = {gb:.1f} GB")
        if gb < 40:
            warn(f"  Local GPU has {gb:.1f} GB — actual training requires L40S 48 GB. "
                 "This smoke test does NOT load the 8B model; it only validates "
                 "the rest of the pipeline.")
    return True


# ── Runner ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bertscore", action="store_true",
                    help="Skip BERTScore inside T7 (saves ~500MB download).")
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache" / "huggingface"),
                    help="Custom HuggingFace cache dir.")
    ap.add_argument("--only", default=None, help="Comma-separated test IDs (T1,T2,…) to run.")
    args = ap.parse_args()

    print(f"\n{CYAN}{'='*72}\n  BRAIN HEALTH pipeline smoke test\n  Folder : {HERE}\n{'='*72}{END}")

    tests = [
        ("T1",  "imports",       t1_imports,                           ()),
        ("T2",  "dorosz_kg",     t2_dorosz_kg,                         ()),
        ("T3",  "dataset_big",   t3_dataset_big,                       (args,)),
        ("T4",  "dataset_small", t4_dataset_small,                     (args,)),
        ("T5",  "tokenizer",     t5_tokenizer_and_prompt,              (args,)),
        ("T6",  "rewards",       t6_rewards,                           ()),
        ("T7",  "metrics",       t7_metrics,                           (args,)),
        ("T8",  "plots",         t8_plots,                             ()),
        ("T9",  "config",        t9_config,                            ()),
        ("T10", "medical",       t10_medical_metrics,                  ()),
        ("T11", "robustness",    t11_robustness,                       ()),
    ]
    only = set(args.only.split(",")) if args.only else None
    summary = []
    for tid, name, fn, fnargs in tests:
        if only and tid not in only:
            continue
        try:
            ok_status = fn(*fnargs)
        except Exception as e:
            fail(f"exception in {tid}: {e}")
            traceback.print_exc()
            ok_status = False
        summary.append((tid, name, ok_status))

    print(f"\n{CYAN}{'='*72}\n  SUMMARY\n{'='*72}{END}")
    for tid, name, st in summary:
        mark = f"{GREEN}PASS{END}" if st else f"{RED}FAIL{END}"
        print(f"  {tid:<3} {mark}  {name}")
    n_pass = sum(1 for _, _, s in summary if s)
    n_fail = sum(1 for _, _, s in summary if not s)
    print(f"\n  {n_pass} passed, {n_fail} failed.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

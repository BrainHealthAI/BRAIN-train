"""robustness_tests.py — Stability assessments for the BRAIN HEALTH LLM.

Two stress tests, both run on a small subset (typ. 30 prompts × 3 perturbations):

  (1) Prompt paraphrasing
      Reformulate the same medical question in 3 different ways. Measure the
      coefficient of variation (CV = std/mean) of BLEU + BS_F1 across the
      paraphrases. A robust model has CV → 0; an unstable one has high CV.

  (2) Simulated ASR noise
      Inject realistic transcription errors into the prompt: char swaps,
      drop chars, insert chars, French-vs-English homophone confusions
      (e.g. "ail" ↔ "ay"). Measure the metric drop vs the clean version.
      For our pipeline ASR → LLM, this quantifies how much the LLM survives
      a noisy transcript.

Both tests share the same model + tokenizer; we re-use them across all
prompts to amortize the (slow) generate() call.
"""
from __future__ import annotations
import random, re, json
from pathlib import Path
import numpy as np

import config as C


# ── (1) Rule-based paraphrasing ──────────────────────────────────────────
# Multilingual rewrites that change surface form without changing meaning.

_PARAPHRASE_REPLACEMENTS = {
    "en": [
        ("What are the symptoms of",  "Could you list the signs of"),
        ("What is",                    "Can you tell me what is"),
        ("How is",                     "How would you describe how"),
        ("treatment for",              "management of"),
        ("How do I",                   "What is the recommended way to"),
        ("Is it",                      "Would it be"),
        ("Should I",                   "Would it be appropriate to"),
        ("What causes",                "What are the causes of"),
        ("Question:",                  "Patient asks:"),
    ],
    "fr": [
        ("Quels sont les symptômes",   "Pourriez-vous lister les signes"),
        ("Qu'est-ce que",              "Que représente"),
        ("Quel est",                   "Quel serait"),
        ("Quel est le traitement",     "Quelle est la prise en charge"),
        ("Comment",                    "De quelle manière"),
        ("Est-ce que",                 "Serait-il possible que"),
        ("Que faire",                  "Quelle conduite à tenir"),
        ("Quels sont les effets",      "Quelles sont les conséquences"),
        ("Question :",                 "Le patient demande :"),
    ],
    "darija": [
        ("شنو هي الأعراض",          "أش كاينين العلامات"),
        ("شنو",                      "أش"),
        ("كيفاش",                    "بأي طريقة"),
        ("واش",                      "هل يمكن"),
        ("السؤال:",                  "المريض كيقول:"),
    ],
}


def paraphrase(prompt: str, lang: str = "en", n: int = 3, seed: int | None = None) -> list[str]:
    """Return up to `n` paraphrases by applying random subsets of rewrites."""
    if seed is not None:
        random.seed(seed)
    rules = _PARAPHRASE_REPLACEMENTS.get(lang, _PARAPHRASE_REPLACEMENTS["en"])
    out, seen = [], {prompt.strip()}
    attempts = 0
    while len(out) < n and attempts < 30:
        attempts += 1
        text = prompt
        # apply 1-3 random rules
        applied = 0
        for src, dst in random.sample(rules, k=min(len(rules), 3)):
            if src in text:
                text = text.replace(src, dst, 1)
                applied += 1
        if applied == 0:
            continue
        if text.strip() in seen:
            continue
        seen.add(text.strip())
        out.append(text)
    # fallback to original if no rule matches
    while len(out) < n:
        out.append(prompt)
    return out[:n]


# ── (2) ASR noise simulation ─────────────────────────────────────────────
# Mimics the kind of errors a Whisper-class ASR makes on French/English/Darija.

_HOMOPHONES_FR = {
    "ai": "é", "et": "ê", "est": "ê", "es": "é", "à": "a", "où": "ou", "ça": "sa",
    "tout": "tou", "ail": "ay", "vert": "ver", "vois": "voit", "sont": "son",
    "non": "no", "ses": "ce", "ces": "se",
}
_HOMOPHONES_EN = {
    "their": "there", "there": "their", "to": "two", "your": "you're",
    "than": "then", "are": "our", "is": "iz", "have": "of", "ate": "eight",
}

def simulate_asr_noise(text: str, lang: str = "en",
                        char_swap_p: float = 0.04,
                        char_drop_p: float = 0.02,
                        homophone_p: float = 0.30,
                        seed: int | None = None) -> str:
    """Return a noisy version of `text` that mimics ASR output."""
    if seed is not None:
        random.seed(seed)
    out_chars = []
    for ch in text:
        roll = random.random()
        if ch.isalpha():
            if roll < char_drop_p:
                continue                               # drop
            if roll < char_drop_p + char_swap_p:
                # swap with neighbour
                neighbour = chr(max(ord('a'), min(ord('z'), ord(ch.lower()) + random.choice([-1, 1]))))
                out_chars.append(neighbour if ch.islower() else neighbour.upper())
                continue
        out_chars.append(ch)
    noisy = "".join(out_chars)
    # homophone substitutions (word-level)
    table = _HOMOPHONES_FR if lang == "fr" else (_HOMOPHONES_EN if lang == "en" else {})
    if table:
        words = noisy.split(" ")
        for i, w in enumerate(words):
            stripped = re.sub(r"[^\wÀ-ÿ]", "", w).lower()
            if stripped in table and random.random() < homophone_p:
                # preserve trailing punctuation
                trail = w[len(stripped):] if w.lower().startswith(stripped) else ""
                words[i] = table[stripped] + trail
        noisy = " ".join(words)
    return noisy


# ── Robustness measurement loop ──────────────────────────────────────────
def measure_robustness(model, tokenizer, eval_rows: list[dict],
                        which: str, sample_n: int = 30,
                        compute_metrics_fn=None,
                        generate_fn=None,
                        seed: int = 0) -> dict:
    """Run the 2 stress tests on `sample_n` prompts.

    `compute_metrics_fn(preds, refs, lang) -> dict` and
    `generate_fn(model, tokenizer, [prompts], max_new) -> [predictions]` come
    from evaluate.py to avoid duplication.

    Returns: {
        "paraphrase": {"BLEU_cv": …, "BS_F1_cv": …, ...},
        "asr_noise":  {"delta_BLEU": …, "delta_BS_F1": …, ...},
        "n_evaluated": int
    }
    """
    from data import build_eval_prompt, SYS_BY_LANG
    assert compute_metrics_fn is not None and generate_fn is not None

    rng = random.Random(seed)
    sample = rng.sample(eval_rows, k=min(sample_n, len(eval_rows)))

    # Group by language
    by_lang: dict[str, list[dict]] = {}
    for r in sample:
        by_lang.setdefault(r["lang"], []).append(r)

    para_cv = {"BLEU": [], "BS_F1": [], "ROUGE_L": []}
    noise_delta = {"BLEU": [], "BS_F1": [], "ROUGE_L": []}
    n_done = 0

    for lang, rows in by_lang.items():
        # Build the canonical user message per row
        for row in rows:
            try:
                msgs, gold = build_eval_prompt(row, lang, which=which)
                base_prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)

                # --- (1) Paraphrase: 3 variants of the user message ---
                user_msg = msgs[-1]["content"]   # the question
                paraphrases = paraphrase(user_msg, lang=lang, n=3, seed=rng.randint(0, 1<<31))
                p_metrics = []
                for ph in paraphrases:
                    msgs_v = list(msgs); msgs_v[-1] = {"role": "user", "content": ph}
                    pr = tokenizer.apply_chat_template(msgs_v, tokenize=False, add_generation_prompt=True)
                    pred = generate_fn(model, tokenizer, [pr], C.EVAL_MAX_NEW_TOK)[0]
                    m = compute_metrics_fn([pred], [gold], lang)
                    p_metrics.append(m)
                # CV across the 3 paraphrases
                for k in para_cv:
                    vals = [pm[k] for pm in p_metrics]
                    mu = float(np.mean(vals))
                    sd = float(np.std(vals))
                    para_cv[k].append((sd / mu) if mu > 1e-6 else 0.0)

                # --- (2) ASR noise: clean vs noisy ---
                clean_pred = generate_fn(model, tokenizer, [base_prompt], C.EVAL_MAX_NEW_TOK)[0]
                noisy_user = simulate_asr_noise(user_msg, lang=lang, seed=rng.randint(0, 1<<31))
                msgs_n = list(msgs); msgs_n[-1] = {"role": "user", "content": noisy_user}
                pr_n = tokenizer.apply_chat_template(msgs_n, tokenize=False, add_generation_prompt=True)
                noisy_pred = generate_fn(model, tokenizer, [pr_n], C.EVAL_MAX_NEW_TOK)[0]
                m_clean = compute_metrics_fn([clean_pred], [gold], lang)
                m_noisy = compute_metrics_fn([noisy_pred], [gold], lang)
                for k in noise_delta:
                    noise_delta[k].append(m_clean[k] - m_noisy[k])
                n_done += 1
            except Exception as e:
                print(f"  [robustness] row skipped: {e}")
                continue

    return {
        "paraphrase":  {f"{k}_cv":      float(np.mean(v) if v else 0.0) for k, v in para_cv.items()},
        "asr_noise":   {f"{k}_delta":   float(np.mean(v) if v else 0.0) for k, v in noise_delta.items()},
        "n_evaluated": n_done,
    }

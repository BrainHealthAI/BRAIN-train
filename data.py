"""Dataset loaders + reasoning-tagged prompt formatting.

Two dataset families:
  - "big"   : Williamsanderson/MedQA-Darija-MultiLingual  (trilingual, +Dorosz KG)
  - "small" : BrainHealthAI/MedQA_mutilangual             (context+question+answer)

Both produce a 'text' column (full chat-template SFT target) and, for SASR,
'prompt_text' (no assistant turn) + 'gold' (the gold answer text only — without
the <answer> tags, used for ROUGE-based reward).

The model is taught to wrap final answers in <answer>...</answer>. SASR's GRPO
composite reward additionally encourages a <think>...</think> block before it.
"""
from __future__ import annotations
from typing import Iterable
from datasets import load_dataset, Dataset, concatenate_datasets

import config as C
from dorosz_kg import get_kg


# ── System prompts ────────────────────────────────────────────────────────
# Enforce: (1) reasoning tags <think>/<answer>, (2) reply in the SAME language
# as the user's question. Both signals are reinforced by the training data
# itself (gold answer is in the same language) and explicitly stated here.
_REASONING_INSTRUCTION_EN = (
    " ALWAYS reply in English (the user's language). "
    "First reason step-by-step inside <think>...</think>, "
    "then provide the final answer inside <answer>...</answer>. "
    "Use only validated medical knowledge; recommend consulting a qualified professional when appropriate."
)
_REASONING_INSTRUCTION_FR = (
    " Répondez TOUJOURS en français (la langue de l'utilisateur). "
    "Raisonnez d'abord étape par étape entre les balises <think>...</think>, "
    "puis donnez la réponse finale entre <answer>...</answer>. "
    "Utilisez uniquement des connaissances médicales validées ; "
    "recommandez de consulter un professionnel qualifié si nécessaire."
)
_REASONING_INSTRUCTION_DR = (
    " جاوب دائماً بالدارجة المغربية (اللغة ديال المستخدم). "
    "فكّر أولاً خطوة بخطوة بين العلامتين <think>...</think>، "
    "ثم أعطِ الجواب النهائي بين <answer>...</answer>. "
    "استعمل فقط معلومات طبية موثوقة، ونصح المريض يمشي لطبيب مختص إذا لزم الأمر."
)

SYS_EN = "You are a careful medical assistant." + _REASONING_INSTRUCTION_EN
SYS_FR = "Vous êtes un assistant médical rigoureux." + _REASONING_INSTRUCTION_FR
SYS_DR = "نتا مساعد طبي دقيق." + _REASONING_INSTRUCTION_DR

SYS_BY_LANG = {"en": SYS_EN, "fr": SYS_FR, "darija": SYS_DR}


def _wrap_answer(ans: str) -> str:
    """Wrap the gold answer in <answer> tags for SFT supervision."""
    return f"{C.ANSWER_OPEN}{ans.strip()}{C.ANSWER_CLOSE}"


def _kg_block(text: str, lang: str) -> str:
    """If `text` (the gold answer or question) mentions Dorosz drugs, return
    a small bullet list to prepend to the user message. Empty string if none."""
    facts = get_kg().lookup_facts(text, max_drugs=3)
    if not facts:
        return ""
    if lang == "darija":
        header = "ملاحظات صيدلية (Dorosz):"
    elif lang == "en":
        header = "Pharmacology notes (Dorosz):"
    else:
        header = "Notes pharmacologiques (Dorosz) :"
    return header + "\n" + "\n".join(facts) + "\n\n"


# ── Big dataset (trilingual, +KG) ─────────────────────────────────────────
def _build_messages_big(row: dict, lang: str, with_kg: bool = True) -> list[dict]:
    q = row[f"question_{lang}"]
    a = row[f"answer_{lang}"]
    kg_prefix = _kg_block(a, lang) if with_kg else ""  # KG sourced from gold (training-time hint)
    if lang == "en":
        user = f"{kg_prefix}Question: {q}"
    elif lang == "fr":
        user = f"{kg_prefix}Question : {q}"
    else:
        user = f"{kg_prefix}السؤال: {q}"
    return [
        {"role": "system",    "content": SYS_BY_LANG[lang]},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": _wrap_answer(a)},
    ]


def _build_prompt_only_big(row: dict, lang: str) -> list[dict]:
    """For inference / GRPO rollout: KG hints are scanned from the QUESTION only
    (we cannot peek at the gold answer at test time)."""
    q = row[f"question_{lang}"]
    kg_prefix = _kg_block(q, lang)
    if lang == "en":
        user = f"{kg_prefix}Question: {q}"
    elif lang == "fr":
        user = f"{kg_prefix}Question : {q}"
    else:
        user = f"{kg_prefix}السؤال: {q}"
    return [
        {"role": "system", "content": SYS_BY_LANG[lang]},
        {"role": "user",   "content": user},
    ]


def _has_trilingual(row: dict) -> bool:
    """Drop rows missing any of the three languages."""
    for l in C.LANGS:
        if not row.get(f"question_{l}") or not row.get(f"answer_{l}"):
            return False
    return True


def load_big(limit: int | None = None, seed: int = 42,
              stratified: bool | None = None) -> Dataset:
    """Load + optionally stratify by specialty_id so each of the ~71 specialties
    gets near-uniform representation. Caps the long tail of dominant specialties
    and forces broader generalization."""
    import random as _random
    if stratified is None:
        stratified = getattr(C, "STRATIFIED_BIG", True)

    ds = load_dataset(C.HF_DATASET_BIG, C.HF_DATASET_BIG_CONF, split="train",
                      cache_dir=str(C.CACHE_DIR))
    # Drop bulky audio columns to save RAM/disk during processing.
    drop = [c for c in ds.column_names if c.startswith("audio_")]
    if drop:
        ds = ds.remove_columns(drop)
    ds = ds.filter(_has_trilingual, num_proc=4)
    ds = ds.shuffle(seed=seed)

    if not limit:
        return ds

    if not stratified:
        return ds.select(range(min(limit, len(ds))))

    # ── Stratified sampling per specialty_id ──────────────────────────
    specs = ds["specialty_id"]
    from collections import defaultdict
    by_spec: dict[str, list[int]] = defaultdict(list)
    for i, sp in enumerate(specs):
        by_spec[sp or "unknown"].append(i)

    n_specs = len(by_spec)
    target_per_spec = max(1, limit // n_specs)
    rng = _random.Random(seed)
    kept: list[int] = []
    short_specs, full_specs = [], []
    for sp, idxs in by_spec.items():
        rng.shuffle(idxs)
        kept.extend(idxs[:target_per_spec])
        if len(idxs) < target_per_spec:
            short_specs.append((sp, len(idxs)))
        else:
            full_specs.append(sp)
    print(f"  [stratified] {n_specs} specialties × ~{target_per_spec} target/spec  "
          f"(short:{len(short_specs)}, full:{len(full_specs)})")

    # Top up with random extras from "full" specs to reach `limit` exactly
    if len(kept) < limit:
        missing = limit - len(kept)
        kept_set = set(kept)
        pool = []
        for sp in full_specs:
            pool.extend(i for i in by_spec[sp] if i not in kept_set)
        rng.shuffle(pool)
        kept.extend(pool[:missing])
    rng.shuffle(kept)
    kept = kept[:limit]
    print(f"  [stratified] final: {len(kept):,} rows kept")
    return ds.select(kept)


# ── Small dataset (context+question+answer, mixed-language) ───────────────
def _norm_lang(s: str | None) -> str:
    if not s:
        return "fr"
    s = s.lower().strip()
    if s.startswith("en"): return "en"
    if s.startswith("fr"): return "fr"
    if s.startswith("ar") or "darij" in s or s == "ma": return "darija"
    return "fr"


def _build_messages_small(row: dict) -> list[dict]:
    """Schema: {question, context_question, answer, language, speciality, article_title}"""
    ctx = (row.get("context_question") or "").strip()
    q   = (row.get("question") or "").strip()
    a   = (row.get("answer") or "").strip()
    lang = _norm_lang(row.get("language"))
    full_q = f"{ctx}, {q}".strip(", ").strip() if ctx else q
    if lang == "en":
        user = f"Question: {full_q}"
    elif lang == "fr":
        user = f"Question : {full_q}"
    else:
        user = f"السؤال: {full_q}"
    return [
        {"role": "system",    "content": SYS_BY_LANG[lang]},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": _wrap_answer(a)},
    ]


def _build_prompt_only_small(row: dict) -> list[dict]:
    msgs = _build_messages_small(row)
    return msgs[:-1]


def load_small(limit: int | None = None, seed: int = 42) -> Dataset:
    """Use both train and test splits as one training pool — we re-split locally
    inside the trainer (the dataset's own 'test' is small and we want max coverage).
    Held-out eval comes from a separate seed-stable shuffle inside evaluate.py."""
    train = load_dataset(C.HF_DATASET_SMALL, C.HF_DATASET_SMALL_CONF, split="train",
                         cache_dir=str(C.CACHE_DIR))
    try:
        test = load_dataset(C.HF_DATASET_SMALL, C.HF_DATASET_SMALL_CONF, split="test",
                            cache_dir=str(C.CACHE_DIR))
        ds = concatenate_datasets([train, test])
    except Exception:
        ds = train
    # Drop empty rows
    ds = ds.filter(lambda r: bool(r.get("question") and r.get("answer")), num_proc=4)
    ds = ds.shuffle(seed=seed)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# ── Public API ────────────────────────────────────────────────────────────
def load_full_dataset(which: str, limit: int | None = None, seed: int = 42) -> Dataset:
    if which == "big":
        return load_big(limit=limit, seed=seed)
    if which == "small":
        return load_small(limit=limit, seed=seed)
    raise ValueError(f"unknown dataset: {which}")


def format_for_sft(ds: Dataset, tokenizer, which: str) -> Dataset:
    """Render each row to a single 'text' column ready for TRL SFTTrainer."""
    if which == "big":
        def _map(row, idx):
            lang = C.LANGS[idx % len(C.LANGS)]
            msgs = _build_messages_big(row, lang, with_kg=True)
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            return {"text": text, "lang": lang}
    else:
        def _map(row, idx):
            msgs = _build_messages_small(row)
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            return {"text": text, "lang": _norm_lang(row.get("language"))}
    cols_to_drop = [c for c in ds.column_names if c not in ("text", "lang")]
    return ds.map(_map, with_indices=True, remove_columns=cols_to_drop, num_proc=4)


def prepare_sasr_dataset(ds: Dataset, tokenizer) -> Dataset:
    """Big-only. Produces text + prompt_text + gold (raw answer, no tags)."""
    def _map(row, idx):
        lang = C.LANGS[idx % len(C.LANGS)]
        full_msgs   = _build_messages_big(row, lang, with_kg=True)
        prompt_msgs = _build_prompt_only_big(row, lang)
        gold = row[f"answer_{lang}"]
        full_text   = tokenizer.apply_chat_template(full_msgs,   tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        return {"text": full_text, "prompt_text": prompt_text, "gold": gold, "lang": lang}
    cols_to_drop = [c for c in ds.column_names if c not in ("text","prompt_text","gold","lang")]
    return ds.map(_map, with_indices=True, remove_columns=cols_to_drop, num_proc=4)


def build_eval_prompt(row: dict, lang: str, which: str) -> tuple[list[dict], str]:
    """For evaluate.py. Returns (messages_without_assistant, gold_answer_text)."""
    if which == "big":
        return _build_prompt_only_big(row, lang), row[f"answer_{lang}"]
    if which == "small":
        return _build_prompt_only_small(row), (row.get("answer") or "")
    raise ValueError(which)

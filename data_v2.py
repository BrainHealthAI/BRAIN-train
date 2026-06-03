"""
Data loaders + chat-template formatters for HELIX-FT v2 (stages A/B/C).

Datasets:
  Stage A — `BrainHealthAI/MedUnified` (20K continued SFT mix)
  Stage B — same MedUnified, subsampled to 3K for SASR
  Stage C — `BrainHealthAI/BrainMedCoT`  (3K from 3458, with <think>...</think>)
"""
from __future__ import annotations

import json
import os
from typing import Optional

from datasets import Dataset, load_dataset

SYSTEM_TPL = {
    "en":     "You are a medical reasoning assistant. Provide a clinically accurate answer in English.",
    "fr":     "Vous êtes un assistant médical. Répondez de manière cliniquement précise en français.",
    "darija": "نتا مساعد طبي. جاوب بطريقة صحيحة طبيا بالدارجة المغربية.",
}

CHAT_TEMPLATE_MEDQA = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
)


def _fmt_sft(row: dict) -> str:
    lang = row.get("language", "fr")
    system = SYSTEM_TPL.get(lang, SYSTEM_TPL["fr"])
    return CHAT_TEMPLATE_MEDQA.format(
        system=system,
        question=row["question"].strip(),
        assistant=row["answer"].strip(),
    )


def _fmt_cot(row: dict) -> str:
    """CoT format: <think>{think}</think>{answer} in the assistant turn."""
    lang = row.get("language", "fr")
    system = SYSTEM_TPL.get(lang, SYSTEM_TPL["fr"]) + " Think step by step inside <think>...</think> before the final answer."
    think = (row.get("think") or "").strip()
    answer = (row.get("answer") or "").strip()
    assistant = f"<think>{think}</think>\n{answer}"
    return CHAT_TEMPLATE_MEDQA.format(
        system=system,
        question=row["question"].strip(),
        assistant=assistant,
    )


def _fmt_prompt_only(row: dict) -> str:
    """For SASR rollouts: prompt up to assistant header, model generates the rest."""
    lang = row.get("language", "fr")
    system = SYSTEM_TPL.get(lang, SYSTEM_TPL["fr"])
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{row['question'].strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ── Public loaders ─────────────────────────────────────────────────────────

def load_medunified_sft(token: Optional[str] = None) -> tuple[Dataset, Dataset]:
    """Stage A: load MedUnified, return (train, eval) HF Datasets with 'text' formatted."""
    tok = token or os.environ.get("HF_TOKEN")
    train = load_dataset("BrainHealthAI/MedUnified", split="train", token=tok) \
        if False else None
    # MedUnified has one parquet per split; use the parquet config:
    train = load_dataset("BrainHealthAI/MedUnified", data_files={"train": "train.parquet",
                                                                 "validation": "eval.parquet"},
                          token=tok)
    train_ds = train["train"].map(lambda r: {"text": _fmt_sft(r)}, remove_columns=[c for c in train["train"].column_names if c != "text"])
    eval_ds = train["validation"].map(lambda r: {"text": _fmt_sft(r)}, remove_columns=[c for c in train["validation"].column_names if c != "text"])
    return train_ds, eval_ds


def load_medunified_sasr_subset(n: int = 3000, token: Optional[str] = None) -> Dataset:
    """Stage B: 3K SASR subset, stratified by language. Returns rows with 'prompt' and 'gold'."""
    tok = token or os.environ.get("HF_TOKEN")
    ds = load_dataset("BrainHealthAI/MedUnified",
                      data_files={"train": "train.parquet"}, token=tok)["train"]
    # Stratify by language: aim 40% EN, 30% FR, 30% Darija (RL needs all 3)
    targets = {"en": int(n * 0.40), "fr": int(n * 0.30), "darija": n - int(n*0.40) - int(n*0.30)}
    by_lang = {l: ds.filter(lambda r, ll=l: r["language"] == ll).shuffle(seed=42)
               for l in ("en", "fr", "darija")}
    pieces = []
    for l, t in targets.items():
        sub = by_lang[l].select(range(min(t, len(by_lang[l]))))
        pieces.append(sub)
    from datasets import concatenate_datasets
    merged = concatenate_datasets(pieces).shuffle(seed=42)
    merged = merged.map(lambda r: {"prompt": _fmt_prompt_only(r), "gold": r["answer"], "lang": r["language"]})
    return merged


def load_brainmedcot_for_cot_stage(n: int = 3000, token: Optional[str] = None) -> tuple[Dataset, Dataset]:
    """Stage C: 3K CoT samples from BrainMedCoT, stratified by source_batch + language."""
    tok = token or os.environ.get("HF_TOKEN")
    ds = load_dataset("BrainHealthAI/BrainMedCoT", token=tok)
    train_full = ds["train"].shuffle(seed=42)
    # Stratify by (source_batch, language): keep proportions
    keep = train_full.select(range(min(n, len(train_full))))
    keep = keep.map(lambda r: {"text": _fmt_cot(r)})
    keep = keep.remove_columns([c for c in keep.column_names if c != "text"])
    eval_set = ds["validation"].map(lambda r: {"text": _fmt_cot(r)})
    eval_set = eval_set.remove_columns([c for c in eval_set.column_names if c != "text"])
    return keep, eval_set

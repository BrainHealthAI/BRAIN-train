# BRAIN-train — `Training_v2` / HELIX-FT v2

## Summary

This repository contains the HELIX-FT v2 training and evaluation workflow: merge the
SFT-Big adapter into Llama 3.1 8B Instruct, then run continued SFT (MedUnified), SASR,
and CoT stages. The published HELIX artifact is an adapter in that model lineage, not
a standalone model to attach directly to an arbitrary base. This is research software;
benchmark scores below are not clinical validation or evidence of safe diagnosis.

Three findings from the available evidence:

1. The pinned HELIX evaluation reports MCQ accuracy from 41.5% to 67.0% across the
   listed English benchmarks and 55.0% on 200 Darija items. These are sampled benchmark
   results, not a matched comparison against Llama-3.1-8B-Instruct.
2. Generation scores are SB_CS 0.695 / 0.630 / 0.581 and ROUGE-L 0.163 / 0.123 / 0.031
   for EN / FR / Darija (100 held-out MedUnified examples per language). They measure
   similarity to reference text, not factual correctness or clinical utility.
3. The reward implementation uses SB_CS as its semantic content component, but the
   published evaluation also reports SB_CS; this shared measure is not independent
   evidence of SASR benefit. No controlled baseline or ablation establishes that benefit.

## Evidence and results

The numbers below are transcribed from the [published `eval_v2.json` at immutable
revision `dd0b813`](https://huggingface.co/BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2/blob/dd0b813bde0e17b3a67fa90c4304a94f1f3c0130/eval_v2.json).
Its timestamp is 2026-05-31. MCQ method is `log_likelihood_argmax`; the JSON records
counts, not a random sampling protocol or confidence intervals. The benchmark loaders
and scoring method are in [`final_eval.py`](final_eval.py). In particular, PubMedQA's
loader uses `pqa_labeled/train` and passes the question without the source abstract, so
this result is not interchangeable with standard context-provided PubMedQA evaluation.

| Benchmark | Accuracy | n |
|---|---:|---:|
| MedQA-USMLE | 52.50% | 200 |
| MedMCQA | 41.50% | 200 |
| PubMedQA (question only; no abstract context) | 53.50% | 200 |
| MMLU clinical knowledge | 61.00% | 200 |
| MMLU medical genetics | 67.00% | 100 |
| MMLU professional medicine | 54.00% | 200 |
| MMLU college medicine | 57.23% | 173 |
| MedQA-Darija-MCQ-500 | 55.00% | 200 |

| MedUnified held-out generation | SB_CS | ROUGE-L | n | Mean output length (characters) |
|---|---:|---:|---:|---:|
| English | 0.6948 | 0.1628 | 100 | 1,890 |
| French | 0.6296 | 0.1228 | 100 | 1,608 |
| Darija | 0.5809 | 0.0311 | 100 | 789 |

The generation method is free-form generation followed by removal of `<think>...</think>`;
the scorer uses multilingual Sentence-BERT cosine (clamped at zero) and ROUGE-L. Neither
metric checks whether a response is medically correct, complete, or safe. The SASR reward
in [`sasr_reward.py`](sasr_reward.py) combines format (weight 0.20), semantic content
(SB_CS, 0.80), and a lexical Dorosz-drug penalty (up to 0.30). That penalty is a narrow
proxy, not a general hallucination detector. Because SB_CS is both optimized and reported,
it cannot by itself demonstrate improvement from SASR.

The comments at the start of `sasr_reward.py` explicitly describe replacing ROUGE-L
because it penalized paraphrasing by SFT-Big. This is the developers' stated rationale,
not a controlled demonstration that ROUGE-L caused the plateau or that the replacement
improved medical reasoning. ROUGE-L remains in the final evaluation.

PubMedQA's 53.50% must be compared with the majority label on the **same 200 items**.
The approximately 55% "always yes" figure cited in review is not established for that
subset by the pinned JSON. A majority baseline from a different 500/1,000-item split
cannot establish whether HELIX underperforms it.

## Model lineage and scope

- **HELIX-v2 (this branch):** SFT-Big LoRA merged into Llama-3.1-8B-Instruct, followed
  by the Training_v2 A/B/C stages. The published artifact is
  [`BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2`](https://huggingface.co/BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2).
  Follow the artifact's base/adapter metadata and this repository's merge chain for
  inference; do not assume the adapter alone is the complete model.
- **BrainMed-8B:** a separate model/research lineage, with a different starting model
  and evaluation setup. Its scores must not be presented as a HELIX baseline or a direct
  head-to-head comparison.
- **Brain Health application:** a separate product/integration layer. The existence of
  this training code or benchmark results does not establish application behavior,
  deployment status, or clinical validation.

The canonical repository is [BrainHealthAI/BRAIN-train](https://github.com/BrainHealthAI/BRAIN-train),
branch [`Training_v2`](https://github.com/BrainHealthAI/BRAIN-train/tree/Training_v2); the
20 October 2026 documentation revision was prepared in
[zadbox/BRAIN-train](https://github.com/zadbox/BRAIN-train). This documentation
describes that branch's pipeline, not every upstream branch or later artifact revision.

## Limitations and evidence gaps

- No matched Llama-3.1-8B-Instruct or SFT-Big result is included in the pinned HELIX JSON.
  Do not infer a gain by comparing with published scores from different splits,
  distributions, prompts, or scoring harnesses.
- The 200 Darija items are only a subset of the named 500-item dataset. An audit of the
  dataset reported empty `source_pair_id` values and a simple answer-choice heuristic
  scoring 132/200 (66%) on the first 200 rows, compared with HELIX's reported 55%.
  On all 500 items, choosing the shortest option alone is correct in 250 cases (50.0%),
  a strong answer-length cue that must be removed or controlled in any rebuilt set.
  The [archived counts and method](evidence/darija_audit_2026-10-07.json) are supplied.
  `final_eval.py` uses the first items, but absent dataset hashes/item IDs, identity with
  the historical HELIX subset is not independently established. Treat provenance and
  heuristic findings as audit-reported, not newly reproduced on the raw dataset;
  they make the benchmark's item independence and style sensitivity open concerns, not
  proof of model capability or incapability.
- The result JSON does not identify exact item IDs, selection seed, or per-item outputs;
  uncertainty intervals and subgroup estimates cannot be reconstructed from it alone.
- Dataset/source quality, translation fidelity, duplicate/near-duplicate cases and
  cross-stage leakage require independent checks. No physician-adjudicated test set or
  clinical outcome evaluation is established here.
- Scores from `evaluate_v2.py` and `final_eval.py` are not interchangeable: the former
  generates/parses MCQ letters, while the latter scores letter-token logits and strips
  reasoning tags for generation. The pinned results identify the latter method.
- Reported ROUGE-L and SB_CS describe textual similarity only. In particular, low
  ROUGE-L does not establish that ROUGE-L was “abandoned” throughout evaluation: the
  current reward no longer uses it, while `final_eval.py` still reports it.
- The README/launch estimates historically stated here are not measured cost, runtime,
  or performance guarantees. Training logs and a controlled stage-by-stage ablation are
  needed to attribute effects to SFT, SASR, or CoT.

## Research priorities

**Short term — measurement and negative results.** Freeze dataset revisions and item
manifests; report per-item paired predictions, sample selection, denominators and
uncertainty. Compare the base, merged SFT-Big, and each HELIX stage on the same cases and
prompt/scoring harness; include no-SASR and reward-component ablations. Publish null or
negative outcomes as readily as gains. Audit overlap across training stages and evaluation
data before interpreting benchmark scores.

**Medium term — paired multilingual, evidence-grounded medical evaluation.** Build an
independent, physician-reviewed set of source cases with linked EN/FR/Darija versions,
preserving case groups across splits and documenting translation review and provenance.
Evaluate correctness, unsupported claims, omissions, uncertainty, language fidelity and
safe escalation against cited evidence; report paired results and subgroup counts. Keep
this distinct from short-term benchmark remeasurement and do not describe it as completed.

Related maintained documentation:
[BrainMed research](https://github.com/BrainHealthAI/BrainMed-8B) and
[application evidence scope](https://github.com/zadbox/helix-brain-health).

## Runnable workflow

See [`LAUNCH_V2.md`](LAUNCH_V2.md) for the historical RunPod workflow and prerequisites.
The scripts expose separate operations: `launch_v2.sh` orchestrates configured training
stages, `evaluate_v2.py` is the generation-based evaluator, `reeval_loglikelihood.py`
provides log-likelihood MCQ scoring, and `final_eval.py` combines log-likelihood MCQ
with per-language generation metrics. They require their dependencies, model access,
datasets and compute; no new run is implied by this documentation. Do not put tokens or
other secrets in tracked files.

Research/educational use only. Not a medical device; outputs require review by qualified
health professionals. Respect the licenses and terms of the base model, adapters, datasets
and dependencies; this repository's license, if any, does not supersede upstream terms.

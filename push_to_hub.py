"""push_to_hub.py — Push the 3 fine-tuned LoRA adapters + plots to HuggingFace.

Models created:
  Williamsanderson/MedQA-Llama3.1-8B-SFT-Big
  Williamsanderson/MedQA-Llama3.1-8B-SASR-Big
  Williamsanderson/MedQA-Llama3.1-8B-SFT-Small

Each gets a model card with:
  - training recipe + hyperparameters
  - macro-avg metrics + delta vs baseline
  - per-language breakdown
  - links to the 5 comparison plots (uploaded with the SASR-Big repo as the canonical home)
"""
import argparse, json, shutil
from pathlib import Path
from huggingface_hub import HfApi, whoami
import config as C


CARD_TEMPLATE = """---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
tags:
- medical
- healthcare
- multilingual
- french
- darija
- arabic
- qlora
- {method_tag}
- brain-health
datasets:
- {dataset_repo}
language:
- en
- fr
- ar
pipeline_tag: text-generation
---

# {model_name}

**Method:** {method_full}
**Base model:** `meta-llama/Llama-3.1-8B-Instruct`
**Training dataset:** `{dataset_repo}` ({dataset_size} samples used){kg_line}

This adapter is one of three trained in parallel for the **BRAIN HEALTH / Operation HELIX-FT** project to compare the impact of (a) dataset size & language coverage and (b) training method (SFT vs SASR) on multilingual medical Q&A.

## Output format

The model is trained to wrap final answers in `<answer>...</answer>` tags. SASR-trained adapters additionally emit a `<think>...</think>` reasoning block before the answer (composite reward incentivizes both).

```text
<think>Brief clinical reasoning here…</think>
<answer>Final answer to the patient/clinician here…</answer>
```

## Training recipe

| Setting | Value |
|---|---|
| Fine-tuning method | {method_full} |
| Parameter-efficient adaptation | **QLoRA** (4-bit NF4 + LoRA rank {lora_r}) |
| Effective batch size | {effective_batch} (per_device={pdb} × grad_accum={gas}) |
| Learning rate | {lr} |
| Epochs | {epochs} |
| Max sequence length | {max_len} |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Anti-overfit | early-stop patience {esp} on eval_loss + cosine LR + warmup ratio {wr} |
| Hardware | RunPod L40S 48 GB |

{sasr_extra}

## Metrics — macro-avg across EN / FR / Darija

{metrics_table}

See `comparison.json` (uploaded with the SASR-Big repo) for per-language breakdown
and `plots/` for paper-style figures comparing all four configurations
(baseline / SFT-Small / SFT-Big / SASR-Big).

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")

# The model is language-locking: question in Darija → answer in Darija, etc.
msgs = [
    {{"role": "system", "content": "Vous êtes un assistant médical rigoureux. Répondez en français…"}},
    {{"role": "user",   "content": "Question : Quels sont les symptômes du diabète de type 2 ?"}},
]
inputs = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Known limitations

- Prototype R&D only — **not a certified medical device.**
- Possible hallucinations despite fine-tuning; not a sole decision aid.
- Darija outputs use Arabic script; quality is uneven across the long tail of
  rare specialties.
- Eval is on a held-out tail of the **big** trilingual dataset for fair
  cross-config comparison; this can disadvantage the small-data adapter on
  Darija (it was trained on a more French-heavy corpus).

## References

- **QLoRA**: Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).
  *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **LoRA**: Hu et al. (2022). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Llama 3**: Grattafiori et al. (2024). [arXiv:2407.21783](https://arxiv.org/abs/2407.21783){sasr_refs}

Part of the **BRAIN HEALTH** project — Operation HELIX-FT, MVP 2026.
"""


def metrics_md_table(comparison: dict, model_key: str) -> str:
    if not comparison:
        return "_Metrics not yet computed._"
    macro = comparison.get("macro", {})
    if not macro:
        return "_Metrics not yet computed._"
    lines = ["| Metric | Baseline | This model | Δ vs base |", "|---|---:|---:|---:|"]
    for m in ["BLEU", "ROUGE_L", "METEOR", "F1", "BS_F1", "SB_CS", "NASS"]:
        b = macro.get(m, {}).get("baseline", 0.0)
        t = macro.get(m, {}).get(model_key, 0.0)
        d = t - b
        sign = "↑" if d > 0 else ("↓" if d < 0 else "=")
        lines.append(f"| **{m}** | {b:.3f} | {t:.3f} | {sign} {d:+.3f} |")
    return "\n".join(lines)


SASR_EXTRA = """
## SASR — composite reward design

This adapter is trained with **Step-wise Adaptive SFT + Reinforcement Learning (SASR)**, starting from the SFT-Big checkpoint. At each optimizer step, the trainer adaptively decides between an SFT update and a GRPO update according to the gradient-norm ratio (Eq. 12 of the paper):

$$ p_t = \\frac{\\| \\nabla \\mathcal{L}_{SFT}^{t} \\|}{\\| \\nabla \\mathcal{L}_{SFT}^{t} \\| + \\gamma \\| \\nabla \\mathcal{L}_{SFT}^{0} \\|} $$

The GRPO reward is **composite** (CVD-thinking style):

| Component | Weight | What it rewards |
|---|---:|---|
| Format reward | 0.30 | Both `<think>…</think>` and `<answer>…</answer>` present (1.0); only `<answer>` (0.5); neither (0.0). |
| Content reward | 0.70 | ROUGE-L between extracted `<answer>` and gold + bonus for medical-JSON cues on complex questions or conciseness on simple questions. |

### SASR hyperparameters

| Setting | Value |
|---|---|
| Warm-up (pure SFT) steps | {warmup} |
| γ (SFT-bias factor) | {gamma} |
| G (rollouts per prompt) | {g_samples} |
| β (KL to SFT-warm reference) | {beta} |
| Clip ε | {clip_eps} |
| Format / content weights | {wf} / {wc} |
"""

SASR_REFS = """
- **SASR**: Chen, J. et al. (2025). *Step-wise Adaptive Integration of SFT and RL for Task-Specific LLMs.*
  [arXiv:2505.13026](https://arxiv.org/abs/2505.13026)
- **GRPO** (used inside SASR): Shao, Z. et al. (2024). *DeepSeekMath.*
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
"""


CONFIGS = [
    {"kind": "sft_big",    "out": C.SFT_BIG_OUT,    "suffix": "SFT-Big",
     "method_full": "Supervised Fine-Tuning (SFT)",
     "dataset_repo": C.HF_DATASET_BIG, "dataset_size": f"{C.BIG_LIMIT:,} (sampled)",
     "with_kg": True, "lr": C.LEARNING_RATE_SFT,  "epochs": C.NUM_EPOCHS},
    {"kind": "sasr_big",   "out": C.SASR_BIG_OUT,  "suffix": "SASR-Big",
     "method_full": "SFT + Step-wise Adaptive RL (SASR — arXiv:2505.13026)",
     "dataset_repo": C.HF_DATASET_BIG, "dataset_size": f"{C.BIG_LIMIT:,} (sampled)",
     "with_kg": True, "lr": C.LEARNING_RATE_SASR, "epochs": C.NUM_EPOCHS},
    {"kind": "sft_small",  "out": C.SFT_SMALL_OUT, "suffix": "SFT-Small",
     "method_full": "Supervised Fine-Tuning (SFT)",
     "dataset_repo": C.HF_DATASET_SMALL, "dataset_size": "68 743 train + 7 639 test",
     "with_kg": False, "lr": C.LEARNING_RATE_SFT, "epochs": C.NUM_EPOCHS},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True, help="HF username (e.g. Williamsanderson)")
    ap.add_argument("--prefix",   default="MedQA-Llama3.1-8B")
    ap.add_argument("--private",  action="store_true")
    args = ap.parse_args()

    api = HfApi(); _ = whoami()

    comparison = {}
    if C.COMPARISON_OUT.exists():
        comparison = json.loads(C.COMPARISON_OUT.read_text())

    for cfg in CONFIGS:
        out_dir = cfg["out"]
        if not out_dir.exists():
            print(f"Skip {cfg['kind']}: no adapter at {out_dir}")
            continue
        repo_id = f"{args.username}/{args.prefix}-{cfg['suffix']}"
        print(f"\n→ Pushing {cfg['kind']} → {repo_id}")
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)

        sasr_extra = ""
        sasr_refs = ""
        if cfg["kind"] == "sasr_big":
            sasr_extra = SASR_EXTRA.format(
                warmup=C.SASR_WARMUP_STEPS, gamma=C.SASR_GAMMA,
                g_samples=C.SASR_G_SAMPLES, beta=C.SASR_BETA,
                clip_eps=C.SASR_CLIP_EPS,
                wf=C.SASR_W_FORMAT, wc=C.SASR_W_CONTENT,
            )
            sasr_refs = SASR_REFS

        card = CARD_TEMPLATE.format(
            model_name=f"{args.prefix}-{cfg['suffix']}",
            method_tag=cfg["kind"],
            method_full=cfg["method_full"],
            dataset_repo=cfg["dataset_repo"],
            dataset_size=cfg["dataset_size"],
            kg_line=" ; **augmented with the Dorosz Causal KG**." if cfg["with_kg"] else "",
            lora_r=C.LORA_R, effective_batch=C.PER_DEVICE_BATCH * C.GRAD_ACCUM_STEPS,
            pdb=C.PER_DEVICE_BATCH, gas=C.GRAD_ACCUM_STEPS,
            lr=cfg["lr"], epochs=cfg["epochs"], max_len=C.MAX_SEQ_LEN,
            esp=C.EARLY_STOP_PATIENCE, wr=C.WARMUP_RATIO,
            sasr_extra=sasr_extra, sasr_refs=sasr_refs,
            metrics_table=metrics_md_table(comparison, cfg["kind"]),
            repo_id=repo_id,
        )
        (out_dir / "README.md").write_text(card, encoding="utf-8")

        # Upload comparison.json + plots only with SASR-Big (canonical home)
        if cfg["kind"] == "sasr_big":
            if C.COMPARISON_OUT.exists():
                shutil.copy(C.COMPARISON_OUT, out_dir / "comparison.json")
            if C.PLOTS_DIR.exists():
                target = out_dir / "plots"
                target.mkdir(exist_ok=True)
                for p in C.PLOTS_DIR.glob("*.png"):
                    shutil.copy(p, target / p.name)
            for extra in ["qualitative_samples.md", "three_lang_demo.json"]:
                src = C.OUTPUTS_DIR / extra
                if src.exists():
                    shutil.copy(src, out_dir / extra)

        api.upload_folder(
            folder_path=str(out_dir), repo_id=repo_id, repo_type="model",
            commit_message=f"Initial {cfg['suffix']} adapter for BRAIN HEALTH",
        )
        print(f"  ✓ https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

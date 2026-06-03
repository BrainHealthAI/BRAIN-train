"""
Generate comparison plots for HELIX-v2 vs published baselines.

Usage on the pod:
    python plot_results.py --eval /workspace/outputs/eval_v2_fixed.json --out-dir /workspace/outputs/plots

Reads our actual numbers from eval_v2_fixed.json and compares against published
baselines (Llama-3.1-8B-Instruct, Meditron-7B, BioMistral-7B). Partial JSON works —
missing benchmarks just won't be plotted.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

# ── Published baselines (zero-shot accuracy, %) ──────────────────────────────
# All cited; mark as published "literature" reference in the plot.
#   Llama-3.1-8B-Instruct: Llama 3 paper (Grattafiori et al. 2024) + community measurements
#   Meditron-7B:           Chen et al. 2023 (arXiv:2311.16079)
#   BioMistral-7B:         Labrak et al. 2024 (arXiv:2402.10373)
BASELINES = {
    # benchmark            random  Llama-3.1-8B  Meditron-7B  BioMistral-7B
    "MedQA-USMLE":          (25.0, 57.0,        52.0,        50.6),
    "MedMCQA":              (25.0, 46.0,        56.5,        46.0),
    "PubMedQA":             (33.3, 74.0,        74.4,        77.5),
    "MMLU-clinical_knowledge":     (25.0, 71.7,        59.6,        59.9),
    "MMLU-medical_genetics":       (25.0, 70.0,        56.0,        64.0),
    "MMLU-professional_medicine":  (25.0, 70.6,        58.5,        61.0),
    "MMLU-college_medicine":       (25.0, 64.7,        55.5,        59.0),
    "Darija-MCQ-500":              (25.0, None,        None,        None),  # no baseline — we're first
}
MODELS = ["Random", "Llama-3.1-8B", "Meditron-7B", "BioMistral-7B", "HELIX-v2"]
COLORS = {"Random": "#aaaaaa", "Llama-3.1-8B": "#5B9BD5",
          "Meditron-7B": "#70AD47", "BioMistral-7B": "#FFC000",
          "HELIX-v2": "#C00000"}


def plot_mcq(helix_mcq, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benches = [b for b in BASELINES if b in helix_mcq or any(b == k for k in helix_mcq)]
    if not benches:
        print("No MCQ benchmarks to plot."); return
    n = len(benches); n_models = len(MODELS); w = 0.16
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(11, n*1.6), 5.5))
    for i, model in enumerate(MODELS):
        ys = []
        for b in benches:
            tup = BASELINES[b]
            random, llama, med, biom = tup
            if model == "Random":  v = random
            elif model == "Llama-3.1-8B":  v = llama
            elif model == "Meditron-7B":   v = med
            elif model == "BioMistral-7B": v = biom
            else:  # HELIX-v2
                rec = helix_mcq.get(b, {})
                v = rec.get("acc", None)
                v = (v * 100.0) if (v is not None) else None
            ys.append(v if v is not None else np.nan)
        ys = np.array(ys, dtype=float)
        offset = (i - (n_models - 1) / 2) * w
        bars = ax.bar(x + offset, np.nan_to_num(ys, nan=0), width=w,
                      label=model, color=COLORS[model],
                      edgecolor="black" if model == "HELIX-v2" else "none",
                      linewidth=1.5 if model == "HELIX-v2" else 0,
                      alpha=1.0 if model == "HELIX-v2" else 0.85)
        for j, (bar, y) in enumerate(zip(bars, ys)):
            if not np.isnan(y):
                ax.text(bar.get_x() + bar.get_width()/2, y + 1.2,
                        f"{y:.1f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold" if model == "HELIX-v2" else "normal")

    ax.set_xticks(x); ax.set_xticklabels(benches, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 90)
    ax.set_title("Medical MCQ Accuracy — HELIX-v2 vs published baselines",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.02,
             "Llama-3.1-8B: Grattafiori et al. 2024 + community measurements · Meditron-7B: Chen et al. 2023 · BioMistral-7B: Labrak et al. 2024",
             ha="center", fontsize=7, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_generation(helix_gen, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = (helix_gen or {}).get("per_language", {})
    if not per:
        print("No generation results to plot."); return
    langs = [l for l in ("en", "fr", "darija") if l in per]
    metrics = ["sb_cs", "rouge_l"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(langs)); w = 0.35
    for i, m in enumerate(metrics):
        ys = [per[l].get(m, 0) for l in langs]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, ys, width=w, label=m.upper(),
                      color="#C00000" if m == "sb_cs" else "#5B9BD5", alpha=0.9)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width()/2, y + 0.01,
                    f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in langs], fontsize=10)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score (0-1)")
    ax.set_title("HELIX-v2 — Generation quality per language (internal eval, n=100)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval_v2_fixed.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.eval, encoding="utf-8") as f:
        data = json.load(f)
    plot_mcq(data.get("mcq", {}), os.path.join(args.out_dir, "comparison_mcq.png"))
    plot_generation(data.get("generation", {}),
                    os.path.join(args.out_dir, "generation_per_lang.png"))

if __name__ == "__main__":
    main()

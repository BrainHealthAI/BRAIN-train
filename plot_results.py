"""plot_results.py — Generate paper-style comparison figures from comparison.json.

Outputs:
  outputs/plots/fig_bleu_comparison.png       (per-model BLEU bars)
  outputs/plots/fig_bertscore_comparison.png  (per-model BS_F1 bars)
  outputs/plots/fig_rouge_l_comparison.png    (per-model ROUGE-L bars)
  outputs/plots/fig_radar_macro.png           (macro-avg radar across all metrics)
  outputs/plots/fig_per_language_heatmap.png  (model × language × metric heatmap)
  outputs/plots/fig_full_table.png            (the 14-column table image, like the paper)

Reads from /workspace/outputs/comparison.json. Assumes evaluate.py has run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import config as C

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans"})

MODEL_COLORS = {
    "baseline":  "#9E9E9E",
    "sft_small": "#FFC107",
    "sft_big":   "#1565C0",
    "sasr_big":  "#C62828",
}
MODEL_LABELS = {
    "baseline":  "Baseline (Llama-3.1-8B)",
    "sft_small": "SFT — small data\n(BrainHealthAI 68K)",
    "sft_big":   "SFT — big data\n(MedQA-Darija 100K + Dorosz)",
    "sasr_big":  "SASR — big data\n(SFT+GRPO 100K + Dorosz)",
}
ALL_METRICS = ["BLEU", "GLEU", "ROUGE_1", "ROUGE_2", "ROUGE_L", "METEOR",
               "Precision", "Recall", "F1",
               "BS_P", "BS_R", "BS_F1",
               "SB_CS", "NASS"]


def _bar(ax, models: list[str], vals: list[float], title: str, ylabel: str):
    bars = ax.bar(range(len(models)), vals,
                  color=[MODEL_COLORS.get(m, "#888") for m in models],
                  edgecolor="black", linewidth=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                       rotation=12, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(0.05, max(vals) * 1.18))


def fig_single_metric(comp: dict, metric: str, out_path: Path):
    macro = comp["macro"][metric]
    models = list(macro.keys())
    vals = [macro[m] for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    _bar(ax, models, vals, f"{metric} — macro-avg across EN / FR / Darija", metric)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def fig_radar(comp: dict, out_path: Path):
    macro = comp["macro"]
    models = list(next(iter(macro.values())).keys())
    metrics = ALL_METRICS
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for m in models:
        vals = [macro[met][m] for met in metrics] + [macro[metrics[0]][m]]
        ax.plot(angles, vals, label=MODEL_LABELS.get(m, m).replace("\n", " "),
                color=MODEL_COLORS.get(m), linewidth=2)
        ax.fill(angles, vals, color=MODEL_COLORS.get(m), alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Macro-avg performance across all metrics", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def fig_per_language_heatmap(comp: dict, out_path: Path):
    per_lang = comp["per_language"]
    langs = list(per_lang.keys())
    models = list(next(iter(per_lang[langs[0]].values())).keys())
    # Use BS_F1 as the headline cell value
    grid = np.zeros((len(models), len(langs)))
    for i, mod in enumerate(models):
        for j, lang in enumerate(langs):
            grid[i, j] = per_lang[lang]["BS_F1"][mod]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=grid.min(), vmax=grid.max())
    ax.set_xticks(range(len(langs))); ax.set_xticklabels([l.upper() for l in langs])
    ax.set_yticks(range(len(models))); ax.set_yticklabels([MODEL_LABELS.get(m, m).replace("\n", " ")
                                                            for m in models], fontsize=9)
    for i in range(len(models)):
        for j in range(len(langs)):
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                    color="white" if grid[i, j] > grid.mean() else "black", fontsize=9)
    ax.set_title("BERTScore-F1 — per language × model")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def fig_full_table(comp: dict, out_path: Path):
    """Render the 14-metric × N-model table as an image (paper-ready)."""
    macro = comp["macro"]
    models = list(next(iter(macro.values())).keys())
    rows = ALL_METRICS
    data = np.array([[macro[m][k] for k in models] for m in rows])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    cell_text = [[f"{v:.3f}" for v in row] for row in data]
    headers = [MODEL_LABELS.get(m, m).replace("\n", " ") for m in models]
    table = ax.table(cellText=cell_text,
                     colLabels=headers, rowLabels=rows,
                     loc="center", cellLoc="center",
                     colColours=[MODEL_COLORS.get(m, "#ddd") for m in models])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    # Highlight best per row
    for i in range(len(rows)):
        best_j = int(np.argmax(data[i]))
        cell = table[(i + 1, best_j)]
        cell.set_facecolor("#FFF59D")
        cell.set_text_props(weight="bold")
    ax.set_title(f"Evaluation metrics — macro across {' / '.join(l.upper() for l in C.LANGS)}\n"
                 f"(highlighted = best per row)", pad=18)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def fig_medical_macro(comp: dict, out_path: Path):
    """4-panel chart of the 4 medical macro-metrics across models."""
    med = comp.get("medical_macro", {})
    if not med:
        print(f"  (no medical metrics — skipping {out_path.name})")
        return
    metrics = ["entity_recall_macro", "hallu_macro", "hit_at_k", "ref_citations_pct"]
    labels  = ["Entity recall (↑ better)", "Hallucination (↓ better)",
               "Diagnostic P@3 (↑ better)", "Cites refs (% rows)"]
    models = list(next(iter(med.values())).keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, mkey, lab in zip(axes.flat, metrics, labels):
        vals = [med[mkey][m] for m in models]
        _bar(ax, models, vals, lab, mkey)
    fig.suptitle("Medical-grade metrics (macro across EN/FR/Darija)",
                 fontsize=14, weight="bold")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def fig_pathology_heatmap(comp: dict, out_path: Path):
    """Per-pathology × model heatmap on entity_recall_macro."""
    by_path = comp.get("medical_per_pathology", {})
    if not by_path:
        print(f"  (no per-pathology data — skipping {out_path.name})")
        return
    cats = list(by_path.keys())
    if not cats:
        return
    models = list(next(iter(by_path.values())).keys())
    grid = np.zeros((len(cats), len(models)))
    for i, c in enumerate(cats):
        for j, m in enumerate(models):
            grid[i, j] = by_path[c].get(m, {}).get("entity_recall_macro", 0.0)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(cats) + 2)))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.05, grid.max()))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models],
                       rotation=12, ha="right", fontsize=9)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=9)
    for i in range(len(cats)):
        for j in range(len(models)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                    color="white" if grid[i,j] > grid.max()/2 else "black", fontsize=8)
    ax.set_title("Entity recall (macro) — per pathology × model", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def fig_robustness(comp: dict, out_path: Path):
    """Two-panel chart: paraphrase BS_F1 CV + ASR-noise ΔBS_F1."""
    rob = comp.get("robustness", {})
    if not rob:
        print(f"  (no robustness data — skipping {out_path.name})")
        return
    models = list(rob.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cv = [rob[m]["paraphrase"]["BS_F1_cv"] for m in models]
    _bar(ax1, models, cv, "Paraphrase stability — BS_F1 CV (↓ better)", "CV(BS_F1)")
    delta = [rob[m]["asr_noise"]["BS_F1_delta"] for m in models]
    _bar(ax2, models, delta, "ASR-noise sensitivity — Δ BS_F1 (↑ more brittle)", "Δ BS_F1")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  ✓ {out_path}")


def main():
    if not C.COMPARISON_OUT.exists():
        print(f"❌ {C.COMPARISON_OUT} not found — run evaluate.py first.")
        return
    comp = json.loads(C.COMPARISON_OUT.read_text())
    out_dir = C.PLOTS_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ Generating plots in {out_dir}")

    fig_single_metric(comp, "BLEU",    out_dir / "fig_bleu_comparison.png")
    fig_single_metric(comp, "BS_F1",   out_dir / "fig_bertscore_comparison.png")
    fig_single_metric(comp, "ROUGE_L", out_dir / "fig_rouge_l_comparison.png")
    fig_single_metric(comp, "F1",      out_dir / "fig_token_f1_comparison.png")
    fig_radar(comp,                   out_dir / "fig_radar_macro.png")
    fig_per_language_heatmap(comp,    out_dir / "fig_per_language_heatmap.png")
    fig_full_table(comp,              out_dir / "fig_full_table.png")
    # Medical-grade panels (skipped silently if data missing)
    fig_medical_macro(comp,           out_dir / "fig_medical_macro.png")
    fig_pathology_heatmap(comp,       out_dir / "fig_pathology_heatmap.png")
    fig_robustness(comp,              out_dir / "fig_robustness.png")

    print(f"\n✓ All plots saved in {out_dir}")


if __name__ == "__main__":
    main()

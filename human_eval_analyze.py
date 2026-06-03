"""human_eval_analyze.py — Aggregate filled human-eval CSVs.

Usage:
  python human_eval_analyze.py reviewer_1.csv reviewer_2.csv ...

Outputs:
  outputs/human_eval_aggregated.json   per-model averages across criteria
  outputs/human_eval_kappa.json        Cohen κ between each annotator pair, per criterion
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import config as C


CRITERIA = ["correctness", "completeness", "safety", "reasoning", "overall"]


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _de_anonymize(rows: list[dict], key_path: Path) -> list[dict]:
    """Resolve A/B/C/D back to model names using the saved key."""
    if not key_path.exists():
        print(f"WARN: anonymisation key {key_path} missing — keeping A/B/C/D labels.")
        return rows
    keymap = {entry["row_id"]: entry["label_to_model"]
              for entry in json.loads(key_path.read_text())}
    out = []
    for row in rows:
        rid = int(row.get("row_id", -1))
        m = keymap.get(rid, {})
        new = dict(row)
        for label in ("A", "B", "C", "D"):
            model = m.get(label, label)
            for crit in CRITERIA:
                old = f"{label}_{crit}_1to5"
                if old in new:
                    new[f"{model}__{crit}"] = new.pop(old)
        out.append(new)
    return out


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Compute Cohen's κ between two lists of categorical 1-5 ratings."""
    assert len(a) == len(b)
    if not a:
        return 0.0
    cats = sorted(set(a) | set(b))
    n = len(a)
    # Observed agreement
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement
    pe = 0.0
    for c in cats:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for x in b if x == c) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="Filled human_eval CSVs (1+ annotators)")
    args = ap.parse_args()

    key_path = C.OUTPUTS_DIR / "human_eval_anonymisation_key.json"

    # Load + de-anonymise per annotator
    annotator_rows: dict[str, list[dict]] = {}
    for p in args.csvs:
        path = Path(p)
        rows = _load_csv(path)
        rows = _de_anonymize(rows, key_path)
        ann = rows[0].get("annotator", path.stem) if rows else path.stem
        annotator_rows[ann] = rows
        print(f"Loaded {len(rows)} rows from {ann}")

    # Aggregate: per model, per criterion, mean across all rows × annotators
    models = set()
    for rows in annotator_rows.values():
        for r in rows:
            for k in r.keys():
                if "__" in k:
                    models.add(k.split("__")[0])

    agg = {m: {c: [] for c in CRITERIA} for m in models}
    for rows in annotator_rows.values():
        for r in rows:
            for m in models:
                for c in CRITERIA:
                    v = _to_float(r.get(f"{m}__{c}"))
                    if v is not None:
                        agg[m][c].append(v)

    summary = {m: {c: (sum(v)/len(v) if v else None) for c, v in cdict.items()}
               for m, cdict in agg.items()}
    print("\n=== Mean scores (1-5, higher better) ===")
    print(f"{'model':<14} " + "  ".join(f"{c:>14}" for c in CRITERIA))
    for m in sorted(summary.keys()):
        print(f"{m:<14} " + "  ".join(
            f"{summary[m][c]:>14.2f}" if summary[m][c] is not None else f"{'-':>14}"
            for c in CRITERIA))

    out = {"means": summary,
            "n_annotators": len(annotator_rows),
            "n_rows_per_annotator": {a: len(r) for a, r in annotator_rows.items()}}
    out_path = C.OUTPUTS_DIR / "human_eval_aggregated.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")

    # Inter-annotator agreement (Cohen κ) on overall score per model
    if len(annotator_rows) >= 2:
        print("\n=== Cohen κ between annotators (overall score per model) ===")
        anns = list(annotator_rows.keys())
        kappa_out: dict = {}
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                a1, a2 = anns[i], anns[j]
                rows1 = {int(r["row_id"]): r for r in annotator_rows[a1] if "row_id" in r}
                rows2 = {int(r["row_id"]): r for r in annotator_rows[a2] if "row_id" in r}
                shared = sorted(set(rows1) & set(rows2))
                pair_key = f"{a1}_vs_{a2}"
                kappa_out[pair_key] = {}
                for m in sorted(models):
                    for c in CRITERIA:
                        v1, v2 = [], []
                        for rid in shared:
                            x = _to_float(rows1[rid].get(f"{m}__{c}"))
                            y = _to_float(rows2[rid].get(f"{m}__{c}"))
                            if x is not None and y is not None:
                                v1.append(int(round(x))); v2.append(int(round(y)))
                        if not v1:
                            continue
                        k = cohen_kappa(v1, v2)
                        kappa_out[pair_key][f"{m}__{c}"] = k
                        if c == "overall":
                            print(f"  {pair_key:<24} {m:<10} κ_overall = {k:.3f}  (n={len(v1)})")
        kp = C.OUTPUTS_DIR / "human_eval_kappa.json"
        kp.write_text(json.dumps(kappa_out, indent=2))
        print(f"\nSaved: {kp}")


if __name__ == "__main__":
    main()

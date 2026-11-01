from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import config
from src.eval import boundary_labels, cp_edit


def text_and_tables(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for label, docs in (metrics.get("results") or {}).items():
        scorable = {k: d for k, d in docs.items()
                    if d.get("gold_clauses") or d.get("gold_metadata") or d.get("gold_tables")}
        text_vals = [d["text_edit"]["edit_dist"] for d in scorable.values() if d.get("text_edit")]
        teds_vals = [d["table_teds"] for d in docs.values() if isinstance(d.get("table_teds"), (int, float))]
        out[label] = {
            "text_edit": statistics.fmean(text_vals) if text_vals else None,
            "table_teds": 100 * statistics.fmean(teds_vals) if teds_vals else None,
            "docs_scored": len(text_vals), "docs_with_tables": len(teds_vals),
        }
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def merge_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in rows if r["outcome"] == "carried")
    fp = sum(1 for r in rows if r["outcome"] == "merge")
    fn = sum(1 for r in rows if r["outcome"] == "split")
    return {"cp_precision": 100 * tp / (tp + fp) if (tp + fp) else None,
            "cp_recall": 100 * tp / (tp + fn) if (tp + fn) else None,
            "cp_f1": 100 * _f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def compute(run_dir: Path, eval_dir: Optional[Path] = None, gold_root: Optional[Path] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    eval_dir = Path(eval_dir) if eval_dir else run_dir / config.EVAL_SUBDIR
    gold_root = Path(gold_root or config.BENCHMARK_GOLD_ROOT)
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    per_model = text_and_tables(metrics)
    if len(per_model) > 1:
        print(f"[paper_metrics] {len(per_model)} extractor models in this run; C-P uses the first model's "
              "predictions of each document.")

    cp = cp_edit.score_run(eval_dir / "final_documents", gold_root)
    docs = boundary_labels.label_run(run_dir, gold_root)
    rows = [r for d in docs for r in d["rows"]]
    with (eval_dir / "boundary_labels.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=boundary_labels.CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in boundary_labels.CSV_FIELDS})

    for label, m in per_model.items():
        m["cp"] = cp["cp_edit"]
        m.update(merge_scores([r for r in rows if r["model"] == label]))
    res = {"run": run_dir.name, "models": per_model,
           "counts": {"cp_docs": cp["counts"]["docs"], "cp_clauses": cp["counts"]["clauses"],
                      "boundary_docs_skipped": sum(1 for d in docs if d["skip_reason"])},
           "per_doc_cp": cp["per_doc"]}
    (eval_dir / "paper_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, m in per_model.items():
        print(f"[paper_metrics] {label}: " + ", ".join(
            f"{k} {m[k]:.3f}" if isinstance(m[k], float) else f"{k} {m[k]}"
            for k in ("text_edit", "table_teds", "cp", "cp_precision", "cp_recall", "cp_f1")))
    return res

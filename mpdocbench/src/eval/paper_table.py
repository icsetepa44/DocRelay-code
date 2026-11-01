"""The MPDocBench-Parse metrics reported in the paper: six columns + Overall, from evaluator results.

    Text Edit, Trunc. Text Edit, Read Order   1 - the evaluator's score (lower is better)
    Table TEDS, Trunc. Table TEDS, Heading TEDS  100 * the evaluator's score (higher is better)
    Overall                                      mean of the six columns, each edit distance counted as (1 - edit) * 100

The values come from the evaluator's own aggregation (mpdocbench._aggregate: mean over documents; reading order
pooled), so several runs (e.g. an English and a Chinese run) can be pooled by concatenating their per-document
results. Formula (CDM) and figure scores are reported by the evaluator but are not part of Overall.

    from src.eval import paper_table
    paper_table.build([run_dir_eng, run_dir_chn])   # -> paper_table.json next to the first run's metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.eval import mpdocbench


PAPER_KEYS = ("text", "merged_text", "table", "merged_table", "reading_order", "heads")


def paper_columns(per_doc: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """The evaluator's own aggregation, mapped to the paper's six columns + Overall.
    A column no doc can score (e.g. no cross-page table in the docs given) is None, and so is Overall then."""
    o = mpdocbench._aggregate(per_doc, {})["overall"]

    def edit(k):
        return None if o[k] is None else 1 - o[k]

    def pct(k):
        return None if o[k] is None else 100 * o[k]

    vals = [o[k] for k in PAPER_KEYS]
    return {"Text Edit ↓": edit("text"), "Trunc. Text Edit ↓": edit("merged_text"),
            "Table TEDS ↑": pct("table"), "Trunc. Table TEDS ↑": pct("merged_table"),
            "Read Order ↓": edit("reading_order"), "Heading TEDS ↑": pct("heads"),
            "Overall ↑": None if any(v is None for v in vals) else sum(100 * v for v in vals) / 6}


def load_per_doc(run_dir: Path) -> List[Dict[str, Any]]:
    """Per-document evaluator results of one run (<run>/_mpdocbench/metrics.json, single extractor model)."""
    metrics = json.loads((Path(run_dir) / "_mpdocbench" / "metrics.json").read_text(encoding="utf-8"))
    (_label, res), = metrics["models"].items()
    return list(res["per_doc"])


def build(run_dirs: Sequence[Path], out_path: Optional[Path] = None) -> Dict[str, Any]:
    """Paper columns + Overall over the pooled documents of run_dirs; written to out_path
    (default <first run>/_mpdocbench/paper_table.json)."""
    run_dirs = [Path(r) for r in run_dirs]
    per_doc = [d for r in run_dirs for d in load_per_doc(r)]
    res = {"runs": [r.name for r in run_dirs], "documents": len(per_doc), "columns": paper_columns(per_doc)}
    out_path = Path(out_path) if out_path else run_dirs[0] / "_mpdocbench" / "paper_table.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    ascii_name = lambda k: k.replace("↓", "").replace("↑", "").strip()  # noqa: E731 (console-safe)
    print("[paper_table] " + ", ".join(f"{ascii_name(k)} {v:.3f}" if isinstance(v, float) else f"{ascii_name(k)} {v}"
                                       for k, v in res["columns"].items()) + f" ({len(per_doc)} documents)")
    return res

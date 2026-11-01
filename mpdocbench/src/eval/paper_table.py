from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.eval import mpdocbench


PAPER_KEYS = ("text", "merged_text", "table", "merged_table", "reading_order", "heads")


def paper_columns(per_doc: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
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
    metrics = json.loads((Path(run_dir) / "_mpdocbench" / "metrics.json").read_text(encoding="utf-8"))
    (_label, res), = metrics["models"].items()
    return list(res["per_doc"])


def build(run_dirs: Sequence[Path], out_path: Optional[Path] = None) -> Dict[str, Any]:
    run_dirs = [Path(r) for r in run_dirs]
    per_doc = [d for r in run_dirs for d in load_per_doc(r)]
    res = {"runs": [r.name for r in run_dirs], "documents": len(per_doc), "columns": paper_columns(per_doc)}
    out_path = Path(out_path) if out_path else run_dirs[0] / "_mpdocbench" / "paper_table.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    ascii_name = lambda k: k.replace("↓", "").replace("↑", "").strip()
    print("[paper_table] " + ", ".join(f"{ascii_name(k)} {v:.3f}" if isinstance(v, float) else f"{ascii_name(k)} {v}"
                                       for k, v in res["columns"].items()) + f" ({len(per_doc)} documents)")
    return res

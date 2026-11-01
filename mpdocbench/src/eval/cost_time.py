"""
Cost and time of one run, summed from the ledger that src/llm/usage.py + run.py write under
``runs/<run>/_cost_time/``: ``llm_calls.jsonl`` (one line per HTTP attempt), ``pairs.jsonl`` (one line per
(doc, model) attempt) and ``sessions.jsonl`` (one line per run.py pipeline session). Writes ``summary.json`` there.

Counting rules:
  * cost = the provider-reported ``usage.cost`` summed over the calls of each doc's KEPT session (the last session
    that finished that doc). Calls of earlier, abandoned attempts (a doc reset on resume) are reported separately as
    ``abandoned_cost_usd``, never mixed in. Calls with no reported cost are counted in ``calls_missing_cost``.
  * wall time = the sum of the sessions' pipeline wall times (each run runs alone, so it is that run's own time).
  * model time per page = the summed attempt latencies of a page's calls (median / p90 over pages).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm import usage


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # a torn last line (killed mid-write) is skipped
    return out


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(x, nd)


def kept_sessions(pairs: List[Dict[str, Any]]) -> Dict[tuple, str]:
    """(doc, label) -> the session whose attempt finished that pair ("done"); the last one wins."""
    kept: Dict[tuple, str] = {}
    for p in pairs:
        if p.get("status") == "done":
            kept[(p.get("doc"), p.get("label"))] = p.get("session")
    return kept


def summarize_records(calls: List[Dict[str, Any]], pairs: List[Dict[str, Any]],
                      sessions: List[Dict[str, Any]], pages_by_doc: Dict[str, int]) -> Dict[str, Any]:
    """Pure: the summary of one run's ledger. ``pages_by_doc`` = pages of each finished doc (run_summary.json)."""
    kept = kept_sessions(pairs)
    kept_calls, abandoned = [], []
    for c in calls:
        key = (c.get("doc"), c.get("label"))
        (kept_calls if key in kept and c.get("session") == kept[key] else abandoned).append(c)

    def cost(rows):
        return sum(float(c["cost_usd"]) for c in rows if c.get("cost_usd") is not None)

    by_role: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0, "failed_attempts": 0,
                                                              "prompt_tokens": 0, "completion_tokens": 0,
                                                              "latency_s": 0.0})
    for c in kept_calls:
        r = by_role[str(c.get("role") or "unknown")]
        r["calls"] += 1
        r["failed_attempts"] += int(c.get("status") != "ok")
        r["cost_usd"] += float(c.get("cost_usd") or 0.0)
        r["prompt_tokens"] += int(c.get("prompt_tokens") or 0)
        r["completion_tokens"] += int(c.get("completion_tokens") or 0)
        r["latency_s"] += float(c.get("latency_s") or 0.0)   # summed attempt latency (model time) of this role
    for r in by_role.values():
        r["cost_usd"] = round(r["cost_usd"], 6)
        r["latency_s"] = round(r["latency_s"], 1)

    page_time: Dict[tuple, float] = defaultdict(float)
    for c in kept_calls:
        if c.get("page") is not None:
            page_time[(c.get("doc"), c.get("label"), c.get("page"))] += float(c.get("latency_s") or 0.0)
    per_page = list(page_time.values())

    done_docs = sorted({doc for doc, _ in kept})
    pages = sum(int(pages_by_doc.get(d, 0)) for d in done_docs)
    total = cost(kept_calls)
    wall = sum(float(s.get("wall_s") or 0.0) for s in sessions)
    doc_time = sum(float(p.get("duration_s") or 0.0) for p in pairs
                   if (p.get("doc"), p.get("label")) in kept and p.get("session") == kept[(p.get("doc"), p.get("label"))])
    return {
        "docs_done": len(done_docs),
        "pages_done": pages,
        "cost_usd": _r(total, 6),
        "cost_per_100_pages_usd": _r(100 * total / pages, 4) if pages else None,
        "abandoned_cost_usd": _r(cost(abandoned), 6),
        "calls": len(kept_calls),
        "failed_attempts": sum(1 for c in kept_calls if c.get("status") != "ok"),
        "calls_missing_cost": sum(1 for c in kept_calls if c.get("status") == "ok" and c.get("cost_usd") is None),
        "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in kept_calls),
        "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in kept_calls),
        "by_role": dict(sorted(by_role.items())),
        "sessions": len(sessions),
        "wall_time_s": _r(wall, 1),
        "doc_time_sum_s": _r(doc_time, 1),
        "model_s_per_page_median": _r(statistics.median(per_page), 2) if per_page else None,
        "model_s_per_page_p90": _r(_pct(per_page, 0.9), 2),
        "wall_s_per_page": _r(wall / pages, 2) if pages else None,
    }


def _pages_by_doc(run_dir: Path) -> Dict[str, int]:
    out = {}
    for rs in run_dir.glob("*/*/run_summary.json"):
        try:
            out[rs.parent.parent.name] = int(json.loads(rs.read_text(encoding="utf-8")).get("pages") or 0)
        except (ValueError, OSError):
            continue
    return out


def summarize(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Write ``<run>/_cost_time/summary.json``; None (nothing written) when the run has no ledger."""
    d = Path(run_dir) / usage.COST_TIME_DIR
    if not d.exists():
        return None
    summary = summarize_records(_read_jsonl(d / usage.CALLS_FILE), _read_jsonl(d / usage.PAIRS_FILE),
                                _read_jsonl(d / usage.SESSIONS_FILE), _pages_by_doc(Path(run_dir)))
    summary = {"run": Path(run_dir).name, **summary}
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8", newline="\n")
    return summary

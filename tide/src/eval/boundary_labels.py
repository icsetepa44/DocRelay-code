from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_DU = Path(__file__).resolve().parents[2]
if str(_DU) not in sys.path:
    sys.path.insert(0, str(_DU))
from src.core import config
from src.eval import adapt
from src.eval.evaluate import (PLACEHOLDER_RE, _table_key, discover_pairs,
                               flatten_table, normalize_prose, tokenize)
from src.ocr import chandra_adapter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

SUSPECT_OVERLAP = 0.2

OUTCOMES = ["carried", "split", "merge", "correct_break", "metadata_span", "page_missing", "gold_gap"]
CSV_FIELDS = ["document", "model", "page_before", "page_after", "gold_label", "pipeline_label", "outcome",
              "joined_as_metadata", "overlap", "suspect", "cause", "carry_symptom", "gold_span", "gold_clauses", "pred_units"]


def _as_int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _gold_item_text(it: Dict[str, Any], table_text: Dict[str, str]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: f" {table_text.get(f'table_{int(m.group(1))}', '')} ",
                              str(it.get("text") or ""))


def gold_items(gold: Dict[str, Any]) -> List[Dict[str, Any]]:
    table_text: Dict[str, str] = {}
    for t in gold.get("tables", []) or []:
        key = _table_key(t.get("table_id"))
        if key and key not in table_text:
            table_text[key] = flatten_table(t)
    out = []
    for i, it in enumerate(gold.get("parsed_text", []) or []):
        ps, pe = _as_int(it.get("page_start")), _as_int(it.get("page_end"))
        if ps is None or pe is None or pe < ps:
            continue
        if str(it.get("metadata_category") or "").strip().lower() == "page":
            continue
        out.append({"type": str(it.get("type") or "").strip().lower(),
                    "number": it.get("clause_number") or it.get("text_id") or f"item_{i}",
                    "ps": ps, "pe": pe, "text": _gold_item_text(it, table_text)})
    for i, t in enumerate(gold.get("tables", []) or []):
        ps, pe = _as_int(t.get("page_start")), _as_int(t.get("page_end"))
        if ps is None or pe is None or pe < ps:
            continue
        out.append({"type": "table", "number": t.get("table_id") or f"table_{i}", "ps": ps, "pe": pe,
                    "text": flatten_table(t)})
    return out


def covered_pages(items: List[Dict[str, Any]]) -> Set[int]:
    return {pg for x in items for pg in range(x["ps"], x["pe"] + 1)}


def gold_total_pages(gold: Dict[str, Any], items: List[Dict[str, Any]]) -> int:
    tp = _as_int((gold.get("document_info") or {}).get("total_pages"))
    return tp if tp else max((x["pe"] for x in items), default=0)


def _unit_pages(unit: Dict[str, Any]) -> Set[int]:
    return {p for p in (adapt.block_page(b) for b in unit.get("source_block_ids", []) or []) if p is not None}


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _read(path: Path) -> Optional[Any]:
    return _load_json(path) if path.exists() else None


def _split_cause(q: int, pages_dir: Path, units: List[Dict[str, Any]]) -> Tuple[str, str]:
    if any({q, q + 1} <= _unit_pages(u) for u in units):
        return "stitched_but_dropped", ""
    ext = _read(pages_dir / f"{q:04d}" / "extraction.json")
    carry = _read(pages_dir / f"{q:04d}" / "carry_state.json")
    rej = _read(pages_dir / f"{q + 1:04d}" / "carry_rejection.json")
    if ext is None or carry is None or rej is None:
        return "unknown", ""
    rej_units = rej.get("units", []) or []
    rejected = [u for u in rej_units if u.get("status") == "rejected"]
    if rejected:
        return "opened_rejected", "+".join(sorted({str(u.get("symptom") or "") for u in rejected}))
    opened = bool(carry.get("open_units"))
    continues = any(d.get("decision") == "continues"
                    for d in ((ext.get("continuation") or {}).get("decisions") or []))
    if opened:
        statuses = sorted({str(u.get("status") or "") for u in rej_units})
        return "opened_closed", "+".join(s for s in statuses if s)
    if not continues:
        return "never_opened", ""
    return "other", ""


def _merge_cause(q: int, members: List[Dict[str, Any]]) -> str:
    return "via_stitch" if any({q, q + 1} <= _unit_pages(u) for u in members) else "via_parent"


def _bag(text: str) -> Counter:
    return Counter(tokenize(normalize_prose(text, drop_tables=False)))


def _containment(a: Counter, b: Counter) -> Optional[float]:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    return sum((a & b).values()) / min(na, nb)


def _coverage(of: Counter, by: Counter) -> Optional[float]:
    n = sum(of.values())
    return sum((of & by).values()) / n if n else None


def label_document(doc: str, model: str, fd_path: Path, gold: Dict[str, Any],
                   ocr_pages: Optional[int]) -> Dict[str, Any]:
    items = gold_items(gold)
    total = gold_total_pages(gold, items)
    base = {"document": doc, "model": model, "pages": total, "rows": [], "skip_reason": None}
    if ocr_pages is not None and ocr_pages != total:
        return {**base, "skip_reason": "page_count_mismatch"}
    if total < 2:
        return {**base, "skip_reason": "single_page"}

    fd = json.loads(Path(fd_path).read_text(encoding="utf-8"))
    units = fd.get("units", []) or []
    by_id = {u["unit_id"]: u for u in units}
    members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for u in units:
        members[adapt._top_ancestor(u["unit_id"], by_id)].append(u)
    pred = [c for c in adapt.adapt_file(Path(fd_path))["contents"] if c.get("page_start") is not None]
    pages_dir = Path(fd_path).parent / "pages"
    have = {p for p in range(total) if (pages_dir / f"{p:04d}" / "extraction.json").exists()}

    clauses = [x for x in items if x["type"] == "clause"]
    spanners = [x for x in items if x["type"] in ("clause", "table")]
    metas = [x for x in items if x["type"] not in ("clause", "table")]
    covered = covered_pages(items)
    bags: Dict[int, Counter] = {}

    def bag_of(obj: Dict[str, Any]) -> Counter:
        if id(obj) not in bags:
            bags[id(obj)] = _bag(obj.get("full_text") if "full_text" in obj else obj["text"])
        return bags[id(obj)]

    rows = []
    for p in range(1, total):
        q = p - 1
        g_span = [c for c in spanners if c["ps"] <= p < c["pe"]]
        p_span = [c for c in pred if c["page_start"] <= p < c["page_end"]]
        if p not in covered or p + 1 not in covered:
            gold_label = "no_gold"
        elif g_span:
            gold_label = "continues"
        elif any(m["ps"] <= p < m["pe"] for m in metas):
            gold_label = "metadata_continues"
        else:
            gold_label = "breaks"
        pipeline_label = "joined" if p_span else "broke"
        row: Dict[str, Any] = {"document": doc, "model": model, "page_before": p, "page_after": p + 1,
                               "gold_label": gold_label, "pipeline_label": pipeline_label,
                               "joined_as_metadata": bool(p_span) and all(c["type"] == "metadata" for c in p_span),
                               "overlap": None, "suspect": False, "cause": "", "carry_symptom": "",
                               "gold_span": max((c["pe"] - c["ps"] for c in g_span), default=None),
                               "gold_clauses": ";".join(str(c["number"]) for c in g_span),
                               "pred_units": ";".join(c["unit_id"] for c in p_span)}
        if q not in have or q + 1 not in have:
            row["outcome"] = "page_missing"
        elif gold_label == "no_gold":
            row["outcome"] = "gold_gap"
        elif gold_label == "metadata_continues":
            row["outcome"] = "metadata_span"
        elif gold_label == "continues":
            if p_span:
                row["outcome"] = "carried"
                ov = [o for o in (_containment(bag_of(pc), bag_of(gc)) for pc in p_span for gc in g_span)
                      if o is not None]
                row["overlap"] = round(max(ov), 4) if ov else None
            else:
                row["outcome"] = "split"
                row["cause"], row["carry_symptom"] = _split_cause(q, pages_dir, units)
        else:
            if p_span:
                row["outcome"] = "merge"
                row["cause"] = _merge_cause(q, [m for pc in p_span for m in members.get(pc["unit_id"], [])])
                ends = [c for c in clauses if c["pe"] == p]
                starts = [c for c in clauses if c["ps"] == p + 1]
                row["gold_clauses"] = ";".join(str(c["number"]) for c in ends + starts)
                if ends and starts:
                    best = None
                    for pc in p_span:
                        ce = [v for v in (_coverage(bag_of(c), bag_of(pc)) for c in ends) if v is not None]
                        cs = [v for v in (_coverage(bag_of(c), bag_of(pc)) for c in starts) if v is not None]
                        if ce and cs:
                            v = min(max(ce), max(cs))
                            best = v if best is None else max(best, v)
                    row["overlap"] = round(best, 4) if best is not None else None
            else:
                row["outcome"] = "correct_break"
        if row["outcome"] in ("carried", "merge") and row["overlap"] is not None:
            row["suspect"] = row["overlap"] < SUSPECT_OVERLAP
        rows.append(row)
    return {**base, "rows": rows}


def _ocr_page_count(doc: str) -> Optional[int]:
    try:
        md = chandra_adapter.ocr_dir_for(doc) / "markdown"
    except (FileNotFoundError, KeyError):
        return None
    n = len(list(md.glob("page_*.md"))) if md.is_dir() else 0
    return n or None


def label_run(run_dir: Path, gold_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    gold_root = Path(gold_root or config.BENCHMARK_GOLD_ROOT)
    pairs, _ = discover_pairs(Path(run_dir), gold_root)
    docs = []
    for doc, model, fd, gold_path in tqdm(pairs, desc="boundary labels", unit="doc"):
        gold = _load_json(gold_path) or {}
        docs.append(label_document(doc, model, fd, gold, _ocr_page_count(doc)))
    return docs

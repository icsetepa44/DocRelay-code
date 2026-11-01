from __future__ import annotations

import functools
import json
import re
import statistics
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.core import config
from src.eval.evaluate import (EXCLUDE_GOLD_METADATA_CATEGORIES, PLACEHOLDER_RE, _levenshtein,
                               _table_key, flatten_table, gold_path_for, normalize_prose)

WORKERS = 8
FURNITURE_RE = re.compile(r"[\d০-৯\s\-–—.,।]*")

EU = Tuple[int, int]


def clean_and_normalize(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        cat = unicodedata.category(ch)
        if cat[0] in "PZC" or cat == "Sk":
            continue
        d = unicodedata.decimal(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


def is_furniture(raw: str) -> bool:
    s = normalize_prose(raw)
    return bool(FURNITURE_RE.fullmatch(s)) or (len(s) < 80 and "গেজেট" in s)


def _int(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def gold_tables(d: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for t in d.get("tables", []) or []:
        k = _table_key(t.get("table_id"))
        if k and k not in out:
            out[k] = flatten_table(t)
    return out


def expand_tables(raw: str, tables: Dict[str, str]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: " " + tables.get(f"table_{int(m.group(1))}", "") + " ", raw)


def gold_clauses(gold_path: Path) -> List[Tuple[str, str]]:
    d = json.loads(gold_path.read_text(encoding="utf-8"))
    tables = gold_tables(d)
    out = []
    for it in d.get("parsed_text", []):
        if str(it.get("type", "")).strip().lower() != "clause":
            continue
        if str(it.get("metadata_category") or "").strip().lower() in EXCLUDE_GOLD_METADATA_CATEGORIES:
            continue
        ps, pe = _int(it.get("page_start")), _int(it.get("page_end"))
        if ps is None or pe is None or pe <= ps:
            continue
        raw = str(it.get("text", ""))
        prose = clean_and_normalize(normalize_prose(PLACEHOLDER_RE.sub(" ", raw)))
        if prose:
            out.append((prose, clean_and_normalize(normalize_prose(expand_tables(raw, tables), drop_tables=False))))
    return out


def pred_items(contents: List[Dict]) -> Tuple[List[str], List[str]]:
    prose_all, full = [], []
    for x in contents:
        furniture = x.get("type") == "metadata"
        p = normalize_prose(x.get("text") or "") or normalize_prose(x.get("full_text") or "", drop_tables=False)
        if p:
            prose_all.append(clean_and_normalize(p))
        f = normalize_prose(x.get("full_text") or "", drop_tables=False)
        if f and not (furniture and is_furniture(f)) and clean_and_normalize(f):
            full.append(clean_and_normalize(f))
    return prose_all, full


def ned(a: str, b: str) -> EU:
    return _levenshtein(a, b), max(len(a), len(b))


def ratio(eu: EU) -> float:
    return eu[0] / eu[1] if eu[1] else 0.0


def best_single(g: str, pred: List[str]) -> List[Tuple[float, int, EU]]:
    found = []
    for j, p in enumerate(pred):
        u = max(len(g), len(p))
        if not u or abs(len(g) - len(p)) / u >= 1.0:
            continue
        e, u = ned(g, p)
        if e < u:
            found.append((e / u, j, (e, u)))
    return sorted(found)


def best_match(g: str, pred: List[str]) -> EU:
    found = best_single(g, pred)
    return found[0][2] if found else (len(g), len(g))


def score_clause(prose: str, expanded: str, prose_all: List[str], full_nf: List[str]) -> EU:
    cands = [best_match(prose, prose_all)]
    if expanded:
        cands.append(best_match(expanded, full_nf))
    return min(cands, key=ratio)


def score_doc(gold_path: Path, contents: List[Dict]) -> List[EU]:
    items = pred_items(contents)
    return [score_clause(p, e, *items) for p, e in gold_clauses(gold_path)]


def score_doc_name(pred_root: Path, gold_root: Path, doc: str) -> Tuple[str, List[EU]]:
    gp = gold_path_for(gold_root, doc)
    if not gp:
        return doc, []
    contents = json.loads(next((pred_root / doc).glob("*/final_stitched_document.json"))
                          .read_text(encoding="utf-8")).get("contents", [])
    return doc, score_doc(gp, contents)


def self_check() -> None:
    assert clean_and_normalize("The  Importer, ১২ days.") == clean_and_normalize("the importer 12 days")
    s = clean_and_normalize("The importer shall pay the duty within thirty days of assessment.")
    assert best_match(s, [s, "x"]) == (0, len(s))


def score_run(pred_root: Path, gold_root: Optional[Path] = None, workers: int = WORKERS) -> Dict[str, Any]:
    self_check()
    pred_root = Path(pred_root)
    gold_root = Path(gold_root or config.BENCHMARK_GOLD_ROOT)
    docs = sorted(p.parent.parent.name for p in pred_root.glob("*/*/final_stitched_document.json"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        per_doc = {d: s for d, s in ex.map(functools.partial(score_doc_name, pred_root, gold_root), docs,
                                            chunksize=4) if s}
    clauses = [eu for s in per_doc.values() for eu in s]
    sums = {d: (sum(e for e, _ in s), sum(u for _, u in s)) for d, s in per_doc.items()}
    return {
        "cp_edit": statistics.fmean(e / u for e, u in sums.values() if u) if sums else None,
        "micro": (sum(e for e, _ in sums.values()) / sum(u for _, u in sums.values())) if sums else None,
        "counts": {"docs": len(per_doc), "clauses": len(clauses),
                   "clauses_ge_0_4": sum(1 for eu in clauses if ratio(eu) >= 0.4),
                   "clauses_lt_0_05": sum(1 for eu in clauses if ratio(eu) < 0.05)},
        "per_doc": {d: {"cp_edit": ratio(sums[d]), "clauses": len(s)} for d, s in per_doc.items()},
    }

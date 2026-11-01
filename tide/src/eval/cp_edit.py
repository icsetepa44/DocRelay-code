"""C-P edit: the cross-page text score of the TIDE results (lower is better).

C-P edit adapts MPDocBench-Parse "Truncated Text Merging" to TIDE. Every gold clause that crosses a page
(page_end > page_start) is compared with the prediction by normalised edit distance, NED = edits / max length (lower
is better). Per document: sum of edits / sum of lengths over its clauses. The score is the mean over documents.

Text on both sides is made lenient first: Unicode NFKC, lower case, every digit -> ASCII, and punctuation, spaces,
invisible and modifier characters dropped.

Matching (best_run): the clause takes its best single predicted item; starting from each of its 3 best items, it
greedily joins the left or right neighbour while the NED keeps falling. The clause keeps the lowest NED of three
comparisons:
  1. clause prose (tables stripped)            vs the predicted items' prose
  2. clause prose (tables stripped)            vs the same, with page furniture removed (predicted metadata items
                                                  that are only digits, or a short "গেজেট" running header)
  3. clause with its tables written out as text vs the predicted items' full_text (tables inline), furniture removed

Input: the adapted predictions that evaluate.evaluate_run writes (<run>/_eval/final_documents/). Entry point:
score_run(), called by src/eval/paper_metrics.py.
"""
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

ROOT = Path(__file__).resolve().parents[2]                 # tide/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.core import config                                 # noqa: E402
from src.eval.evaluate import (EXCLUDE_GOLD_METADATA_CATEGORIES, PLACEHOLDER_RE, _levenshtein,  # noqa: E402
                               _table_key, flatten_table, gold_path_for, normalize_prose)

N_STARTS = 3                                                # best single items tried as seeds for joining
WORKERS = 8
FURNITURE_RE = re.compile(r"[\d০-৯\s\-–—.,।]*")

EU = Tuple[int, int]                                        # (edits, upper)


# ------------------------------------------------------------------ text
def lenient(text: str) -> str:
    """NFKC, lower case, every Unicode digit -> ASCII, drop punctuation / spaces / invisible / modifier chars."""
    out = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        cat = unicodedata.category(ch)
        if cat[0] in "PZC" or cat == "Sk":
            continue
        d = unicodedata.decimal(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


def is_furniture(raw: str) -> bool:
    """A page number (digits and punctuation only) or a short gazette running header."""
    s = normalize_prose(raw)
    return bool(FURNITURE_RE.fullmatch(s)) or (len(s) < 80 and "গেজেট" in s)


def _int(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ inputs
def gold_tables(d: Dict) -> Dict[str, str]:
    """table_N -> the table flattened to text (the first table wins when an id repeats)."""
    out: Dict[str, str] = {}
    for t in d.get("tables", []) or []:
        k = _table_key(t.get("table_id"))
        if k and k not in out:
            out[k] = flatten_table(t)
    return out


def expand_tables(raw: str, tables: Dict[str, str]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: " " + tables.get(f"table_{int(m.group(1))}", "") + " ", raw)


def gold_clauses(gold_path: Path) -> List[Tuple[str, str]]:
    """(prose, expanded) of every gold clause that crosses a page, both lenient; prose = tables stripped,
    expanded = tables written out as text."""
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
        prose = lenient(normalize_prose(PLACEHOLDER_RE.sub(" ", raw)))
        if prose:
            out.append((prose, lenient(normalize_prose(expand_tables(raw, tables), drop_tables=False))))
    return out


def pred_items(contents: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    """The predicted items in document order, lenient, as three lists:
    all prose (an item whose lenient text is empty is kept, and a joined run stops at it);
    prose without furniture; full_text (tables inline) without furniture. Prose falls back to full_text."""
    prose_all, prose, full = [], [], []
    for x in contents:
        furniture = x.get("type") == "metadata"
        p = normalize_prose(x.get("text") or "") or normalize_prose(x.get("full_text") or "", drop_tables=False)
        if p:
            prose_all.append(lenient(p))
            if not (furniture and is_furniture(p)) and lenient(p):
                prose.append(lenient(p))
        f = normalize_prose(x.get("full_text") or "", drop_tables=False)
        if f and not (furniture and is_furniture(f)) and lenient(f):
            full.append(lenient(f))
    return prose_all, prose, full


# ------------------------------------------------------------------ matching
def ned(a: str, b: str) -> EU:
    return _levenshtein(a, b), max(len(a), len(b))


def ratio(eu: EU) -> float:
    return eu[0] / eu[1] if eu[1] else 0.0


def best_single(g: str, pred: List[str]) -> List[Tuple[float, int, EU]]:
    """All single items that beat 'nothing predicted', as (ned, index, (edits, upper)), best first."""
    found = []
    for j, p in enumerate(pred):
        u = max(len(g), len(p))
        if not u or abs(len(g) - len(p)) / u >= 1.0:
            continue
        e, u = ned(g, p)
        if e < u:
            found.append((e / u, j, (e, u)))
    return sorted(found)


def best_run(g: str, pred: List[str]) -> EU:
    """Best single item, then greedily join neighbouring items (left or right) while the NED keeps falling."""
    seeds = best_single(g, pred)[:N_STARTS]
    best = seeds[0][2] if seeds else (len(g), len(g))
    for _, j, eu in seeds:
        lo = hi = j
        cur = eu
        while True:
            step = None
            for a, b in ([(lo - 1, hi)] if lo > 0 else []) + ([(lo, hi + 1)] if hi < len(pred) - 1 else []):
                e, u = ned(g, "".join(pred[a:b + 1]))
                if e / u < ratio(cur) and (step is None or e / u < ratio(step[2])):
                    step = (a, b, (e, u))
            if step is None:
                break
            lo, hi, cur = step
        if ratio(cur) < ratio(best):
            best = cur
    return best


def score_clause(prose: str, expanded: str, prose_all: List[str], prose_nf: List[str], full_nf: List[str]) -> EU:
    """L5: the lowest NED of the three comparisons (the first one wins a tie)."""
    cands = [best_run(prose, prose_all), best_run(prose, prose_nf)]
    if expanded:
        cands.append(best_run(expanded, full_nf))
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


# ------------------------------------------------------------------ checks
def self_check() -> None:
    assert lenient("The  Importer, ১২ days.") == lenient("the importer 12 days")
    halves = ["theimportershallpaytheduty", "withinthirtydaysofassessment"]
    assert best_run("".join(halves), halves) == (0, len("".join(halves))), "a split clause is joined back"


# ------------------------------------------------------------------ run
def score_run(pred_root: Path, gold_root: Optional[Path] = None, workers: int = WORKERS) -> Dict[str, Any]:
    """C-P edit over every document in pred_root (<run>/_eval/final_documents/): the per-document mean, the
    pooled (micro) value, counts and per-document scores. Documents without a crossing gold clause are skipped."""
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

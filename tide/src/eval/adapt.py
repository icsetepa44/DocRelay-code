"""
Eval bridge: convert the pipeline's FinalDocument into the evaluator's prediction shape.

    {"contents": [{"type": "clause"|"metadata"|"table", "text": <prose>, "full_text": <prose + tables>,
                   "unit_id": <top-level unit>, "page_start": int|None, "page_end": int|None}],
     "tables":   [{headers, rows} | {markdown/html}, + "parent_index" into contents],
     "adapter_stats": {table_fragments, recovered_fragments, unrouted_fragments}}

One `contents` item per TOP-LEVEL unit, with every descendant (sub-points, embedded tables) inlined in
document order — gold's notion of a "clause" is a main point together with its sub-points. Hierarchy
comes from the pipeline's parent_unit_id, so the grouping is faithful, not lossy. Document order = the
(page, seq) of each fragment's source block, across the unit and all its descendants: a stitched unit that
continues on page n+1 after its sub-points on page n is written after those sub-points, not before them.

  - text_block unit                                   -> "clause"
  - heading / image_or_seal / header_footer_page_number -> "metadata"
  - a top-level unit that is ONLY a table              -> "table" (scored by table/doc-level metrics only)
  - noise_or_separator                                 -> dropped
  - A3: a "clause" whose source blocks are PURELY furniture skills -> "metadata"

Fragments are routed by CONTENT SHAPE (FIX-1b): a table grid / table block is table text, everything
else is prose. `text` holds prose only (clause metrics); `full_text` adds the inline table text
(doc-level WER / coverage). Every table is also emitted as a structured `tables[]` entry for TEDS/GriTS.
FIX-3: prose falls back text -> description -> markdown -> html, table text grid -> markdown -> html ->
table-as-string -> text, so no extracted text is silently dropped. (FIX-n tags: see evaluate.py.)

Page provenance: `page_start`/`page_end` are the first/last page (1-BASED, the gold convention) of any
source block of the item or its descendants, parsed from the block ids (`{doc}:p{page:04d}:b{seq:04d}`,
0-based); None when no block id carries a page. The evaluator ignores them (it reads type/text/full_text).

Pure functions; reads only the FinalDocument and its sibling pages/*/extraction.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.stages._common import table_struct

# A3: skills that mark a block as non-operative document furniture.
FURNITURE_SKILL_IDS = {"document_metadata", "figure_or_seal"}

_BLOCK_PAGE_RE = re.compile(r":p(\d{4}):b\d{4}$")


def block_page(block_id: str) -> Optional[int]:
    """0-based pipeline page of a block id `{doc}:p{page:04d}:b{seq:04d}` (None if not that shape)."""
    m = _BLOCK_PAGE_RE.search(str(block_id or ""))
    return int(m.group(1)) if m else None


_BLOCK_PAGE_SEQ_RE = re.compile(r":p(\d{4}):b(\d{4})$")


def _page_seq(block_id: Optional[str]) -> Optional[tuple]:
    """(page, seq) of a block id `{doc}:p{page:04d}:b{seq:04d}`, as assembler._page_seq (None if not that shape)."""
    m = _BLOCK_PAGE_SEQ_RE.search(str(block_id or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


# ============================================================
# FRAGMENT ROUTING
# ============================================================

def _grid_text(content: Dict[str, Any]) -> str:
    """A table grid flattened exactly like src.stages._common.table_to_text (empty cells kept)."""
    tbl = content.get("table")
    parts = [str(h) for h in tbl.get("headers", [])]
    for row in tbl.get("rows", []):
        parts.extend(str(c) for c in row)
    return " ".join(parts).strip()


def _table_str(content: Dict[str, Any]) -> str:
    """A `table` value emitted as a raw HTML/markdown STRING instead of a grid ('' otherwise)."""
    t = content.get("table")
    return t.strip() if isinstance(t, str) else ""


def _has_any_text(content: Dict[str, Any]) -> bool:
    return any(str(content.get(k) or "").strip()
               for k in ("text", "description", "markdown", "html", "table"))


def route_fragment(block_type: str, content: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """FIX-1b + FIX-3: split one fragment into prose text and table text, by content shape.

    Returns {"prose", "table", "recovered"}; `recovered` marks text that only the markdown/html/table
    fallbacks found (an earlier adapter dropped it).
    """
    content = content or {}
    has_grid = isinstance(content.get("table"), dict)
    if block_type == "table":
        if has_grid:
            table = _grid_text(content)
        else:
            table = str(content.get("markdown") or content.get("html") or _table_str(content)
                        or content.get("text") or "").strip()
        recovered = bool(table) and not has_grid and not (content.get("markdown") or content.get("html"))
        return {"prose": "", "table": table, "recovered": recovered}
    primary = str(content.get("text") or content.get("description") or "")
    prose = primary or str(content.get("markdown") or content.get("html") or "")
    table = _grid_text(content) if has_grid else _table_str(content)  # a table inside a text_block
    recovered = (not primary.strip()) and bool(prose.strip() or table)
    return {"prose": prose.strip(), "table": table, "recovered": recovered}


def _fragments(unit: Dict[str, Any]):
    """Yield (block_type, content) for a unit, or for each fragment of a stitched (cross-page) unit."""
    content = unit.get("content")
    if isinstance(content, list):
        for frag in content:
            yield frag.get("block_type", ""), frag.get("content") or {}
    else:
        yield unit.get("block_type", ""), content or {}


def _is_table_fragment(block_type: str, content: Dict[str, Any]) -> bool:
    return block_type == "table" or isinstance(content.get("table"), dict) or bool(_table_str(content))


def _table_entry(contents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One structured table entry for a unit, merging its cross-page fragments.

    Structured fragments win (headers from the first, rows concatenated in reading order); otherwise
    joined markdown/html. A `table` value that is a raw string is moved to html/markdown so TEDS can
    parse it (FIX-3). None if empty.
    """
    fixed = []
    for c in contents:
        s = _table_str(c)
        if s and not (c.get("html") or c.get("markdown")):
            c = {**c, ("html" if s.startswith("<") else "markdown"): s}
        fixed.append(c)
    structs = [s for s in (table_struct(c) for c in fixed) if s is not None]
    if structs:
        rows: List[Any] = []
        for s in structs:
            rows.extend(s["rows"])
        return {"headers": structs[0]["headers"], "rows": rows}
    md = "\n".join(str(c.get("markdown")) for c in fixed if c.get("markdown"))
    html = "\n".join(str(c.get("html")) for c in fixed if c.get("html"))
    entry: Dict[str, Any] = {}
    if md:
        entry["markdown"] = md
    if html:
        entry["html"] = html
    return entry or None


# ============================================================
# FINAL DOCUMENT -> PREDICTION
# ============================================================

def _top_ancestor(uid: str, by_id: Dict[str, Any]) -> str:
    """Walk parent_unit_id to the top-level ancestor (guards against cycles)."""
    seen = set()
    cur = by_id[uid]
    while cur.get("parent_unit_id") and cur["parent_unit_id"] in by_id and cur["unit_id"] not in seen:
        seen.add(cur["unit_id"])
        cur = by_id[cur["parent_unit_id"]]
    return cur["unit_id"]


def load_block_skills(pages_dir: Path) -> Dict[str, set]:
    """block_id -> skill ids, from the persisted pages/*/extraction.json (A3).

    The assembler drops the model's per-block skills_used from final_document.json; it survives in the
    page extractions, joined back on source_block_ids <-> block_id.
    """
    out: Dict[str, set] = {}
    if not pages_dir.is_dir():
        return out
    for ext in sorted(pages_dir.glob("*/extraction.json")):
        try:
            pe = json.loads(ext.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for b in pe.get("blocks", []):
            if b.get("block_id"):
                out[b["block_id"]] = {s.get("skill_id") for s in (b.get("skills_used") or [])
                                      if s.get("skill_id")}
    return out


def adapt_final_document(fd: Dict[str, Any], block_skills: Optional[Dict[str, set]] = None) -> Dict[str, Any]:
    """FinalDocument dict -> prediction dict (see the module docstring for the shape)."""
    block_skills = block_skills or {}
    units: List[Dict[str, Any]] = fd.get("units", [])
    by_id = {u["unit_id"]: u for u in units}

    buckets: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    stats = Counter()
    for ui, u in enumerate(units):
        top = _top_ancestor(u["unit_id"], by_id)
        if top not in buckets:
            buckets[top] = {"frags": [], "skills": set(), "pages": set(), "last_key": (-1, -1)}
            order.append(top)
        b = buckets[top]
        sources = u.get("source_block_ids", []) or []
        stitched = isinstance(u.get("content"), list)
        for fi, (bt, c) in enumerate(_fragments(u)):
            r = route_fragment(bt, c)
            # Reading-order key: the fragment's own source block (a stitched unit's fragments line up 1:1 with
            # its source_block_ids); a fragment with no parsable id stays right after its predecessor.
            sid = (sources[fi] if fi < len(sources) else None) if stitched else (sources[0] if sources else None)
            key = _page_seq(sid) or b["last_key"]
            b["last_key"] = key
            b["frags"].append((key, ui, fi, r["prose"], "\n".join(x for x in (r["prose"], r["table"]) if x)))
            if r["table"]:
                stats["table_fragments"] += 1
            if r["recovered"]:
                stats["recovered_fragments"] += 1
            if bt != "noise_or_separator" and not (r["prose"] or r["table"]) and _has_any_text(c):
                stats["unrouted_fragments"] += 1
        for bid in sources:
            b["skills"] |= block_skills.get(bid, set())
            pg = block_page(bid)
            if pg is not None:
                b["pages"].add(pg)

    contents: List[Dict[str, Any]] = []
    index_by_top: Dict[str, int] = {}
    for top in order:
        block_type = by_id[top].get("block_type", "")
        if block_type == "noise_or_separator":
            continue
        b = buckets[top]
        # Fragments of the unit and all its descendants in reading order, then joined run by run (a run = the
        # consecutive fragments of one unit), exactly as before when nothing moves.
        runs: List[List[tuple]] = []
        for f in sorted(b["frags"], key=lambda f: (f[0], f[1], f[2])):
            if runs and runs[-1][-1][1] == f[1]:
                runs[-1].append(f)
            else:
                runs.append([f])
        run_prose = ["\n".join(f[3] for f in run if f[3]).strip() for run in runs]
        run_full = ["\n".join(f[4] for f in run).strip() for run in runs]
        full = "\n".join(t for t in run_full if t).strip()
        if not full:
            continue
        prose = "\n".join(t for t in run_prose if t).strip()
        if not prose:
            utype = "table"  # FIX-1b: a table-only top-level unit is a table, not a metadata item
        else:
            utype = "clause" if block_type == "text_block" else "metadata"
            # A3: union over ALL descendants + non-empty guard, so a real clause carrying any
            # non-furniture skill (or no attribution) is never demoted.
            if utype == "clause" and b["skills"] and b["skills"] <= FURNITURE_SKILL_IDS:
                utype = "metadata"
        index_by_top[top] = len(contents)
        pages = b["pages"]
        contents.append({"type": utype, "text": prose, "full_text": full, "unit_id": top,
                         "page_start": min(pages) + 1 if pages else None,
                         "page_end": max(pages) + 1 if pages else None})

    tables: List[Dict[str, Any]] = []
    for u in units:
        tcs = [c for bt, c in _fragments(u) if _is_table_fragment(bt, c)]
        if not tcs:
            continue
        entry = _table_entry(tcs)
        if entry is None:
            continue
        # parent_index: the clause the table hangs off (None if that ancestor was dropped).
        entry["parent_index"] = index_by_top.get(_top_ancestor(u["unit_id"], by_id))
        tables.append(entry)

    return {"contents": contents, "tables": tables, "adapter_stats": dict(stats)}


def adapt_file(fd_path: Path) -> Dict[str, Any]:
    """Adapt one runs/<run>/<doc>/<model>/final_document.json (reads its sibling pages/ for A3)."""
    fd_path = Path(fd_path)
    fd = json.loads(fd_path.read_text(encoding="utf-8"))
    return adapt_final_document(fd, load_block_skills(fd_path.parent / "pages"))

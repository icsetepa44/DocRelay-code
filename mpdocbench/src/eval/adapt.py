"""
Flattening helpers shared with the MPDocBench evaluation adapter (src/eval/mpdocbench.py): unit text with child
tables inlined, table-content iteration/merging, and the parent_unit_id top-ancestor walk.

flatten_final_document() converts FinalDocument.units into a slim prediction shape ({"contents":[{text,type}]}):
  - text_block unit                -> type "clause"
  - heading / image_or_seal / header_footer_page_number / table -> type "metadata"
  - noise_or_separator             -> dropped
  - a table unit that is a CHILD of a text_block unit is inlined into that clause's text.
Every table is also emitted as a STRUCTURED entry in a top-level `tables` list ({headers, rows | markdown/html}
+ `parent_index` into `contents`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `src.*` importable whether run via run.py or standalone.
_DU = Path(__file__).resolve().parents[2]  # mpdocbench/
if str(_DU) not in sys.path:
    sys.path.insert(0, str(_DU))
from src.stages._common import table_struct, table_to_text  # noqa: E402

# ============================================================
# GLOBALS (edit)
# ============================================================

# A3: skills that mark a block as non-operative document furniture. A top-level
# text_block "clause" whose source blocks are PURELY these skills (per the model's
# own skills_used attribution) is demoted to metadata. See flatten_final_document.
FURNITURE_SKILL_IDS = {"header_footer_page_number"}


# ============================================================
# FLATTENING
# ============================================================

def _content_text(block_type: str, content: Dict[str, Any]) -> str:
    content = content or {}
    if block_type == "table":
        return table_to_text(content)  # shared flatten (src.stages._common)
    return str(content.get("text") or content.get("description") or "")


def _iter_table_contents(unit: Dict[str, Any]):
    """Yield the table content dict(s) inside a unit: the unit itself when it is a table, or
    every table fragment of a stitched (cross-page) table unit."""
    content = unit.get("content")
    if isinstance(content, list):
        for frag in content:
            if frag.get("block_type") == "table":
                yield frag.get("content") or {}
    elif unit.get("block_type") == "table":
        yield content or {}


def _merge_table_contents(contents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One structured table entry for a table unit, merging its cross-page fragments.

    Structured fragments (content["table"]) win: headers from the first, rows concatenated in
    reading order. If none are structured, fall back to joined markdown/html. None if empty.
    """
    structs = [s for s in (table_struct(c) for c in contents) if s is not None]
    if structs:
        rows: List[Any] = []
        for s in structs:
            rows.extend(s["rows"])
        return {"headers": structs[0]["headers"], "rows": rows}
    md = "\n".join(str(c.get("markdown")) for c in contents if c.get("markdown"))
    html = "\n".join(str(c.get("html")) for c in contents if c.get("html"))
    entry: Dict[str, Any] = {}
    if md:
        entry["markdown"] = md
    if html:
        entry["html"] = html
    return entry or None


def _unit_text(unit: Dict[str, Any]) -> str:
    content = unit.get("content")
    if isinstance(content, list):
        return "\n".join(
            _content_text(frag.get("block_type", ""), frag.get("content") or {})
            for frag in content
        ).strip()
    return _content_text(unit.get("block_type", ""), content or {}).strip()


def _unit_type(block_type: str) -> str:
    return "clause" if block_type == "text_block" else "metadata"


def _top_ancestor(uid: str, by_id: Dict[str, Any]) -> str:
    """Walk parent_unit_id to the top-level ancestor (guards against cycles)."""
    seen = set()
    cur = by_id[uid]
    while cur.get("parent_unit_id") and cur["parent_unit_id"] in by_id \
            and cur["unit_id"] not in seen:
        seen.add(cur["unit_id"])
        cur = by_id[cur["parent_unit_id"]]
    return cur["unit_id"]


def _load_block_skills(pages_dir: Path) -> Dict[str, set]:
    """block_id -> set of skill_id, from the persisted page extractions (LLM-free).

    A3 needs the model's per-block skills_used, which the assembler drops from
    final_document.json. It survives in the sibling pages/XXXX/extraction.json, so we
    read it back and join on source_block_ids <-> block_id.
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
            bid = b.get("block_id")
            if bid:
                out[bid] = {s.get("skill_id") for s in (b.get("skills_used") or []) if s.get("skill_id")}
    return out


def flatten_final_document(
    fd: Dict[str, Any], block_skills: Optional[Dict[str, set]] = None
) -> Dict[str, Any]:
    """
    One prediction item per TOP-LEVEL unit, with every descendant (sub-points,
    embedded tables) inlined in document order. This matches gold's notion of a
    "clause" = a main point together with its sub-points/tables. Hierarchy is
    preserved by the pipeline via parent_unit_id, so this grouping is faithful, not lossy.

    A3: when block_skills is supplied, a text_block "clause" whose source blocks are
    PURELY furniture (skills subset of FURNITURE_SKILL_IDS) is demoted to metadata.
    """
    block_skills = block_skills or {}
    units: List[Dict[str, Any]] = fd.get("units", [])
    by_id = {u["unit_id"]: u for u in units}

    buckets: Dict[str, List[str]] = {}
    bucket_skills: Dict[str, set] = {}
    order: List[str] = []
    for u in units:
        top = _top_ancestor(u["unit_id"], by_id)
        if top not in buckets:
            buckets[top] = []
            bucket_skills[top] = set()
            order.append(top)
        buckets[top].append(_unit_text(u))
        for bid in u.get("source_block_ids", []):
            bucket_skills[top] |= block_skills.get(bid, set())

    contents = []
    content_index_by_top: Dict[str, int] = {}
    for top in order:
        block_type = by_id[top].get("block_type", "")
        if block_type == "noise_or_separator":
            continue
        text = "\n".join(t for t in buckets[top] if t).strip()
        if not text:
            continue
        utype = _unit_type(block_type)
        # A3: demote a text_block clause whose source blocks are purely furniture.
        # Union over ALL descendants + non-empty guard => a real clause carrying any
        # non-furniture skill (e.g. numbered_point) or no attribution is never demoted,
        # protecting clause recall.
        skills = bucket_skills[top]
        if utype == "clause" and skills and skills <= FURNITURE_SKILL_IDS:
            utype = "metadata"
        content_index_by_top[top] = len(contents)
        contents.append({"text": text, "type": utype})

    # T3: additionally emit every table as a STRUCTURED entry hanging off its parent rule.
    # Additive only — `contents` above is untouched, so all existing metrics are identical;
    # `parent_index` is the position in `contents` of the table's top-level ancestor (the
    # clause it belongs to), or None if that ancestor was dropped as noise/empty.
    tables: List[Dict[str, Any]] = []
    for u in units:
        table_contents = list(_iter_table_contents(u))
        if not table_contents:
            continue
        entry = _merge_table_contents(table_contents)
        if entry is None:
            continue
        entry["parent_index"] = content_index_by_top.get(_top_ancestor(u["unit_id"], by_id))
        tables.append(entry)

    return {"contents": contents, "tables": tables}


# ============================================================
# RUN
# ============================================================

def latest_run_dir(runs_root: Path) -> Path:
    runs = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {runs_root}")
    return max(runs, key=lambda p: p.name)

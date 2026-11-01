from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_DU = Path(__file__).resolve().parents[2]
if str(_DU) not in sys.path:
    sys.path.insert(0, str(_DU))
from src.stages._common import table_struct, table_to_text


FURNITURE_SKILL_IDS = {"header_footer_page_number"}


def _content_text(block_type: str, content: Dict[str, Any]) -> str:
    content = content or {}
    if block_type == "table":
        return table_to_text(content)
    return str(content.get("text") or content.get("description") or "")


def _iter_table_contents(unit: Dict[str, Any]):
    content = unit.get("content")
    if isinstance(content, list):
        for frag in content:
            if frag.get("block_type") == "table":
                yield frag.get("content") or {}
    elif unit.get("block_type") == "table":
        yield content or {}


def _merge_table_contents(contents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    seen = set()
    cur = by_id[uid]
    while cur.get("parent_unit_id") and cur["parent_unit_id"] in by_id \
            and cur["unit_id"] not in seen:
        seen.add(cur["unit_id"])
        cur = by_id[cur["parent_unit_id"]]
    return cur["unit_id"]


def _load_block_skills(pages_dir: Path) -> Dict[str, set]:
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
        skills = bucket_skills[top]
        if utype == "clause" and skills and skills <= FURNITURE_SKILL_IDS:
            utype = "metadata"
        content_index_by_top[top] = len(contents)
        contents.append({"text": text, "type": utype})

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


def latest_run_dir(runs_root: Path) -> Path:
    runs = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {runs_root}")
    return max(runs, key=lambda p: p.name)

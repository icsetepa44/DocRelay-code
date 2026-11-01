from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.stages._common import table_struct

FURNITURE_SKILL_IDS = {"document_metadata", "figure_or_seal"}

_BLOCK_PAGE_RE = re.compile(r":p(\d{4}):b\d{4}$")


def block_page(block_id: str) -> Optional[int]:
    m = _BLOCK_PAGE_RE.search(str(block_id or ""))
    return int(m.group(1)) if m else None


_BLOCK_PAGE_SEQ_RE = re.compile(r":p(\d{4}):b(\d{4})$")


def _page_seq(block_id: Optional[str]) -> Optional[tuple]:
    m = _BLOCK_PAGE_SEQ_RE.search(str(block_id or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _grid_text(content: Dict[str, Any]) -> str:
    tbl = content.get("table")
    parts = [str(h) for h in tbl.get("headers", [])]
    for row in tbl.get("rows", []):
        parts.extend(str(c) for c in row)
    return " ".join(parts).strip()


def _table_str(content: Dict[str, Any]) -> str:
    t = content.get("table")
    return t.strip() if isinstance(t, str) else ""


def _has_any_text(content: Dict[str, Any]) -> bool:
    return any(str(content.get(k) or "").strip()
               for k in ("text", "description", "markdown", "html", "table"))


def route_fragment(block_type: str, content: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
    table = _grid_text(content) if has_grid else _table_str(content)
    recovered = (not primary.strip()) and bool(prose.strip() or table)
    return {"prose": prose.strip(), "table": table, "recovered": recovered}


def _fragments(unit: Dict[str, Any]):
    content = unit.get("content")
    if isinstance(content, list):
        for frag in content:
            yield frag.get("block_type", ""), frag.get("content") or {}
    else:
        yield unit.get("block_type", ""), content or {}


def _is_table_fragment(block_type: str, content: Dict[str, Any]) -> bool:
    return block_type == "table" or isinstance(content.get("table"), dict) or bool(_table_str(content))


def _table_entry(contents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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


def _top_ancestor(uid: str, by_id: Dict[str, Any]) -> str:
    seen = set()
    cur = by_id[uid]
    while cur.get("parent_unit_id") and cur["parent_unit_id"] in by_id and cur["unit_id"] not in seen:
        seen.add(cur["unit_id"])
        cur = by_id[cur["parent_unit_id"]]
    return cur["unit_id"]


def load_block_skills(pages_dir: Path) -> Dict[str, set]:
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
            utype = "table"
        else:
            utype = "clause" if block_type == "text_block" else "metadata"
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
        entry["parent_index"] = index_by_top.get(_top_ancestor(u["unit_id"], by_id))
        tables.append(entry)

    return {"contents": contents, "tables": tables, "adapter_stats": dict(stats)}


def adapt_file(fd_path: Path) -> Dict[str, Any]:
    fd_path = Path(fd_path)
    fd = json.loads(fd_path.read_text(encoding="utf-8"))
    return adapt_final_document(fd, load_block_skills(fd_path.parent / "pages"))

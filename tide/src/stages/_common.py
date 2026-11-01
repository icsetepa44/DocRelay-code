from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

MAX_TEXT_CHARS = 16000


def clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def payload_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def mixed_ocr_text(markdown: str) -> str:
    text = markdown or ""
    if "<table" not in text.lower():
        return text
    from bs4 import BeautifulSoup
    from bs4.element import Tag

    soup = BeautifulSoup(text, "html.parser")
    parts: List[str] = []
    for node in soup.contents:
        if isinstance(node, Tag) and node.name == "table":
            parts.append("\n" + str(node).strip() + "\n")
        else:
            parts.append(str(node))
    return "".join(parts).strip()


def page_text_payload(page: Any, fmt: str) -> Dict[str, str]:
    if fmt not in ("markdown", "html", "both", "mixed"):
        fmt = "both"
    if fmt == "mixed":
        return {"markdown": clip(mixed_ocr_text(page.markdown))}
    out: Dict[str, str] = {}
    if fmt in ("markdown", "both"):
        out["markdown"] = clip(page.markdown)
    if fmt in ("html", "both"):
        out["html"] = clip(page.html)
    return out


_FORMAT_GUIDANCE = {
    "markdown": "Chandra Markdown (prose and tables both in Markdown).",
    "html": "Chandra HTML.",
    "both": "provided in two forms — Chandra Markdown prose AND full HTML.",
    "mixed": "Chandra Markdown prose with any tables kept inline as HTML.",
}


def format_guidance(fmt: str) -> str:
    return _FORMAT_GUIDANCE.get(fmt, _FORMAT_GUIDANCE["both"])


def skill_candidate_view(record: Any) -> Dict[str, Any]:
    return {
        "skill_id": record.skill_id,
        "version": record.version,
        "description": record.description,
        "applicability": list(record.applicability),
    }


def table_to_text(content: Dict[str, Any]) -> str:
    content = content or {}
    tbl = content.get("table")
    if isinstance(tbl, dict):
        parts = [str(h) for h in tbl.get("headers", [])]
        for row in tbl.get("rows", []):
            parts.extend(str(c) for c in row)
        return " ".join(parts).strip()
    return str(content.get("markdown") or content.get("html") or "").strip()


def table_struct(content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = content or {}
    tbl = content.get("table")
    if isinstance(tbl, dict):
        return {"headers": list(tbl.get("headers", [])),
                "rows": [list(r) for r in tbl.get("rows", [])]}
    return None


def block_text(block: Any) -> str:
    content = getattr(block, "content", None) or {}
    text = str(content.get("text") or content.get("description") or "").strip()
    if text:
        return text
    return table_to_text(content)

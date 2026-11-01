from __future__ import annotations

import json
import re
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


_LABEL_TO_REGION_TYPE = {
    "Section-Header": "heading",
    "Text": "text",
    "Image": "figure", "Figure": "figure", "Diagram": "figure",
    "Table": "table",
    "Equation-Block": "equation",
    "Caption": "caption",
    "List-Group": "list",
    "Page-Header": "header", "Page-Footer": "footer", "Footnote": "footnote",
    "Table-Of-Contents": "toc",
    "Form": "form",
    "Complex-Block": "complex",
}

_LAYOUT_SNIPPET_CHARS = 160
_LAYOUT_INDENT_DIV = 25
_LAYOUT_YGAP = 20

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _layout_snippet(content: Any) -> str:
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", str(content or ""))).strip()
    return text[:_LAYOUT_SNIPPET_CHARS]


def layout_regions(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    layout = layout or {}
    pb = layout.get("page_box") or [0, 0, 0, 0]
    if len(pb) < 4:
        return []
    pw, ph = (pb[2] - pb[0]), (pb[3] - pb[1])
    if not pw or not ph:
        return []
    out: List[Dict[str, Any]] = []
    for chunk in layout.get("chunks") or []:
        bb = chunk.get("bbox") or []
        if len(bb) < 4:
            continue
        box = [max(0, min(1000, round((bb[i] - pb[i % 2]) / (pw if i % 2 == 0 else ph) * 1000)))
               for i in range(4)]
        out.append({
            "region_id": f"L{len(out) + 1:03d}",
            "type": _LABEL_TO_REGION_TYPE.get(chunk.get("label"), "other"),
            "box": box,
            "text": _layout_snippet(chunk.get("content")),
        })
    return out


def layout_payload(layout: Dict[str, Any]) -> str:
    regions = layout_regions(layout)
    if not regions:
        return ""
    lines = ["Page layout (reading order; boxes 0-1000, origin top-left; indentation ~ x-position):"]
    prev_y1: Optional[int] = None
    for r in regions:
        x0, y0, x1, y1 = r["box"]
        if prev_y1 is not None and (y0 - prev_y1) > _LAYOUT_YGAP:
            lines.append("")
        indent = " " * (x0 // _LAYOUT_INDENT_DIV)
        lines.append(f"{indent}[{r['type']} {x0},{y0}-{x1},{y1}] {r['text']}")
        prev_y1 = y1
    return clip("\n".join(lines))


_LAYOUT_NOTE_PRESENT = (
    "a `layout` field is also provided — Chandra's detected regions in reading order, each tagged with a "
    "type and a 0-1000 bounding box and laid out with indentation/blank lines that approximate the page's "
    "2D structure. Use it to decide region boundaries, types, and reading order."
)
_LAYOUT_NOTE_ABSENT = "no layout is provided; infer regions from the page text and image."


def layout_guidance(has_layout: bool) -> str:
    return _LAYOUT_NOTE_PRESENT if has_layout else _LAYOUT_NOTE_ABSENT


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

"""Shared helpers for stage user-prompt construction."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Cap very long OCR text sent to the model (defensive; benchmark pages are small).
MAX_TEXT_CHARS = 16000


def clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def payload_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def mixed_ocr_text(markdown: str) -> str:
    """Chandra markdown as 'prose = Markdown, tables = HTML', in reading order.

    Chandra's markdown already interleaves markdown prose with raw ``<table>`` HTML blocks
    (it never emits pipe tables). This pass makes that invariant explicit and robust: it uses
    an HTML parser to isolate every ``<table>`` block, re-serialises it as well-formed HTML
    (auto-closing any malformed/unclosed table — the win over a naive regex), and keeps all
    other content (prose, headings, inline ``<sup>`` …) verbatim. Reading order is preserved
    by walking the parsed nodes left to right, and each table is set off by blank lines so the
    model sees a clean table island.

    Pure-prose pages (the common case) short-circuit and are returned unchanged.
    """
    text = markdown or ""
    if "<table" not in text.lower():
        return text
    # Local import: bs4 is only needed for the (minority) pages that contain a table.
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
    """OCR text dict for a page filtered by fmt ('markdown' | 'html' | 'both' | 'mixed').

    Returns only the requested format(s), each clipped. 'mixed' returns a single 'markdown'
    key holding markdown prose with tables kept inline as HTML (see ``mixed_ocr_text``).
    Unknown fmt falls back to 'both' so a bad value never silently drops all OCR text.
    """
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


# Per-format guidance injected into stage prompts via the {{ocr_format_note}} token, so each
# prompt describes the OCR representation actually sent by page_text_payload (above). Prevents a
# stage from asking the model for a field (e.g. html) that the chosen format never provides —
# the mixed-format `"html": null` schema crash (FAILURE_ANALYSIS_V8).
_FORMAT_GUIDANCE = {
    "markdown": "Chandra Markdown (prose and tables both in Markdown).",
    "html": "Chandra HTML.",
    "both": "provided in two forms — Chandra Markdown prose AND full HTML.",
    "mixed": "Chandra Markdown prose with any tables kept inline as HTML.",
}


def format_guidance(fmt: str) -> str:
    """Short note describing the OCR representation for `fmt`, for the `{{ocr_format_note}}` token.

    Reads naturally after "current page text: ". Unknown fmt -> 'both' (matches page_text_payload).
    """
    return _FORMAT_GUIDANCE.get(fmt, _FORMAT_GUIDANCE["both"])


# --------------------------------------------------------------------------- layout payload
# Chandra layout `label` -> a coarse region-type hint for the region analyzer's retrieval query.
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

_LAYOUT_SNIPPET_CHARS = 160   # per-region text kept in the layout payload
_LAYOUT_INDENT_DIV = 25       # 0-1000 x0 -> up to ~40 columns of leading indentation
_LAYOUT_YGAP = 20             # blank line when the vertical gap to the previous region exceeds this (of 1000)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _layout_snippet(content: Any) -> str:
    """A region's content -> short single-line text: HTML tags stripped, whitespace collapsed, clipped."""
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", str(content or ""))).strip()
    return text[:_LAYOUT_SNIPPET_CHARS]


def layout_regions(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chandra ``layout`` -> reading-ordered typed region list with boxes normalized to int 0-1000.

    Mirrors the eval side's normalization (``src/eval/mpdocbench.py``): chunk bboxes are pixels in the
    ``page_box`` frame, so we subtract the page-box origin and scale by its width/height to 0-1000.
    Chandra chunks are already in reading order, so list order is preserved. Returns ``[]`` on a
    missing/degenerate ``page_box`` (zero width/height) or no usable chunks; a chunk with fewer than 4
    bbox coords is skipped.
    """
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
    """LAPDoc-style 'SpatialFormat' rendering of the page layout, for the region analyzer.

    Reading-ordered, one line per typed region: ``[type x0,y0-x1,y1] snippet``. Horizontal position is
    encoded as leading indentation (``x0 // _LAYOUT_INDENT_DIV`` spaces) and vertical gaps as blank
    lines, so the model sees the 2D structure reconstructed with whitespace (no boxes painted on the
    image; no reliance on a grounding model). Empty string when there is no usable layout.
    """
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


# Note for the {{layout_note}} region-analyzer token (mirrors _FORMAT_GUIDANCE / format_guidance).
_LAYOUT_NOTE_PRESENT = (
    "a `layout` field is also provided — Chandra's detected regions in reading order, each tagged with a "
    "type and a 0-1000 bounding box and laid out with indentation/blank lines that approximate the page's "
    "2D structure. Use it to decide region boundaries, types, and reading order."
)
_LAYOUT_NOTE_ABSENT = "no layout is provided; infer regions from the page text and image."


def layout_guidance(has_layout: bool) -> str:
    """Short note for the `{{layout_note}}` region-analyzer token (mirrors ``format_guidance``)."""
    return _LAYOUT_NOTE_PRESENT if has_layout else _LAYOUT_NOTE_ABSENT


def skill_candidate_view(record: Any) -> Dict[str, Any]:
    """Router-facing candidate view: id/version/description/applicability (no instructions)."""
    return {
        "skill_id": record.skill_id,
        "version": record.version,
        "description": record.description,
        "applicability": list(record.applicability),
    }


def table_to_text(content: Dict[str, Any]) -> str:
    """Flatten a table block's content to plain text: headers + all row cells, space-joined.

    Falls back to the block's ``markdown``/``html`` when the extractor emitted the
    unstructured variant. Shared by the carry path (so an open table carries real text
    across a page break — T6) and the eval bridge (``adapt``) so the flatten idiom lives in
    exactly one place.

    Every cell is joined verbatim — including EMPTY ones (an empty cell becomes an empty
    string in the join, i.e. an extra space). This is deliberate: it keeps the eval bridge's
    inlined table text byte-identical to the pre-refactor flatten, so the additive ``tables[]``
    slot (T3) provably does not move any metric. Do not filter empty cells here.
    """
    content = content or {}
    tbl = content.get("table")
    if isinstance(tbl, dict):
        parts = [str(h) for h in tbl.get("headers", [])]
        for row in tbl.get("rows", []):
            parts.extend(str(c) for c in row)
        return " ".join(parts).strip()
    return str(content.get("markdown") or content.get("html") or "").strip()


def table_struct(content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Structured ``{headers, rows}`` for a table block, or None when it isn't a grid.

    Returns None for the ``markdown``/``html`` fallback shape (nothing to structure) so a
    caller can decide how to represent it.
    """
    content = content or {}
    tbl = content.get("table")
    if isinstance(tbl, dict):
        return {"headers": list(tbl.get("headers", [])),
                "rows": [list(r) for r in tbl.get("rows", [])]}
    return None


def block_text(block: Any) -> str:
    """A block's text (image description / flattened table cells), stripped — '' if none.

    Shared by carry_builder (the still-open unit's carried text) and the orchestrator
    (carry-rejection analysis) so the extraction idiom lives in exactly one place. Table
    blocks store their content under ``table``/``markdown``/``html`` rather than ``text``,
    so fall back to the flattened table text — otherwise a continuing table would carry
    nothing across a page break (T6).
    """
    content = getattr(block, "content", None) or {}
    text = str(content.get("text") or content.get("description") or "").strip()
    if text:
        return text
    return table_to_text(content)

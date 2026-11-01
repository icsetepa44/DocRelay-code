from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core import config
from src.contracts.page import PageData


def ocr_dir_for(doc_name: str) -> Path:
    root = config.BENCHMARK_OCR_PARENT
    flat = root / doc_name
    if flat.is_dir():
        return flat
    nested = root / config.domain_for(doc_name) / doc_name
    return nested if nested.is_dir() else flat


def _page_image_src(doc_name: str, page_index: int) -> str:
    return str(ocr_dir_for(doc_name) / "page_images" / f"page_{page_index:04d}.png")


def _read_layout(doc_name: str, page_index: int) -> Dict[str, Any]:
    path = ocr_dir_for(doc_name) / "layout" / f"page_{page_index:04d}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _is_doc_dir(p: Path) -> bool:
    return (p / "markdown").is_dir() or (p / f"{p.name}.json").exists()


def list_available_documents() -> List[str]:
    parent = config.BENCHMARK_OCR_PARENT
    if not parent.exists():
        return []
    doc_dirs = []
    for p in sorted(parent.iterdir()):
        if not p.is_dir():
            continue
        if _is_doc_dir(p):
            doc_dirs.append(p)
        else:
            doc_dirs.extend(q for q in sorted(p.iterdir()) if q.is_dir() and _is_doc_dir(q))
    out, unfinished = [], []
    for p in doc_dirs:
        (out if (p / f"{p.name}.json").exists() else unfinished).append(p.name)
    if unfinished:
        print(f"Skipping {len(unfinished)} OCR dir(s) with no completion marker: {unfinished}", flush=True)
    return sorted(out)


def _page_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def _record(doc_name: str, page_number: int, markdown: str, html: str,
            layout: Dict[str, Any], page_image_src: str) -> Dict[str, Any]:
    return {
        "document_id": doc_name,
        "page_number": page_number,
        "source_file": page_image_src,
        "page_image_src": page_image_src,
        "layout": layout or {},
        "content": {"raw_markdown": markdown, "raw_html": html},
    }


def read_page_records(doc_name: str) -> List[Dict[str, Any]]:
    folder = ocr_dir_for(doc_name)
    if not folder.exists():
        raise FileNotFoundError(f"No OCR folder for document '{doc_name}': {folder}")

    md_dir = folder / "markdown"
    html_dir = folder / "html"
    records: List[Dict[str, Any]] = []

    md_files = sorted(md_dir.glob("page_*.md"), key=_page_index) if md_dir.is_dir() else []
    if md_files:
        for md_path in md_files:
            n = _page_index(md_path)
            markdown = md_path.read_text(encoding="utf-8")
            html_path = html_dir / f"page_{n:04d}.html"
            html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
            records.append(_record(doc_name, n, markdown, html,
                                   _read_layout(doc_name, n), _page_image_src(doc_name, n)))
        records.sort(key=lambda r: int(r["page_number"]))
        return records

    marker = folder / f"{doc_name}.json"
    if marker.exists():
        data = json.loads(marker.read_text(encoding="utf-8"))
        for pg in data.get("pages", []):
            n = int(pg.get("page_num", 0))
            layout = {"page_num": n, "page_box": pg.get("page_box"),
                      "chunks": pg.get("chunks") or []}
            records.append(_record(doc_name, n,
                                   str(pg.get("markdown", "") or ""),
                                   str(pg.get("html", "") or ""),
                                   layout, _page_image_src(doc_name, n)))
        records.sort(key=lambda r: int(r["page_number"]))
        if records:
            return records

    raise FileNotFoundError(
        f"No chandra OCR pages found for '{doc_name}' in {folder} "
        f"(looked for markdown/page_*.md and {doc_name}.json)"
    )


def build_page_data(record: Dict[str, Any], page_image: str) -> PageData:
    content = record.get("content") or {}
    return PageData(
        document_id=str(record.get("document_id", "")),
        page_number=int(record.get("page_number", 0)),
        page_image=str(page_image),
        markdown=str(content.get("raw_markdown", "") or ""),
        html=str(content.get("raw_html", "") or ""),
        layout=dict(record.get("layout") or {}),
    )

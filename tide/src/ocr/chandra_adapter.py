from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core import config
from src.contracts.page import PageData


_OCR_DIR_CACHE: Dict[str, Path] = {}

def _build_ocr_cache() -> Dict[str, Path]:
    global _OCR_DIR_CACHE
    if _OCR_DIR_CACHE:
        return _OCR_DIR_CACHE

    search_roots = [config.BENCHMARK_OCR_PARENT]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and ((p / "markdown").is_dir() or (p / f"{p.name}.json").exists()):
                _OCR_DIR_CACHE[p.name] = p
                _OCR_DIR_CACHE[p.name.rstrip(".")] = p
                _OCR_DIR_CACHE[p.name.replace("-", "_")] = p
    return _OCR_DIR_CACHE


def ocr_dir_for(doc_name: str) -> Path:
    direct = config.BENCHMARK_OCR_PARENT / doc_name
    if direct.exists() and ((direct / "markdown").is_dir() or (direct / f"{doc_name}.json").exists()):
        return direct
    
    cache = _build_ocr_cache()
    if doc_name in cache:
        return cache[doc_name]
    if doc_name.rstrip(".") in cache:
        return cache[doc_name.rstrip(".")]
    if doc_name.replace("-", "_") in cache:
        return cache[doc_name.replace("-", "_")]

    return direct


def _source_pdf(doc_name: str) -> str:
    clean_name = doc_name.rstrip(".")
    candidates = [
        config.BENCHMARK_PDF_ROOT / f"{doc_name}.pdf",
        config.BENCHMARK_PDF_ROOT / f"{clean_name}.pdf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(config.BENCHMARK_PDF_ROOT / f"{clean_name}.pdf")


def list_available_documents() -> List[str]:
    cache = _build_ocr_cache()
    return sorted({p.name for p in cache.values()})


def _page_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def _record(doc_name: str, page_number: int, markdown: str, html: str) -> Dict[str, Any]:
    return {
        "document_id": doc_name,
        "page_number": page_number,
        "source_file": _source_pdf(doc_name),
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
            records.append(_record(doc_name, n, markdown, html))
        records.sort(key=lambda r: int(r["page_number"]))
        return records

    marker = folder / f"{doc_name}.json"
    if marker.exists():
        data = json.loads(marker.read_text(encoding="utf-8"))
        for pg in data.get("pages", []):
            n = int(pg.get("page_num", 0))
            records.append(_record(doc_name, n,
                                   str(pg.get("markdown", "") or ""),
                                   str(pg.get("html", "") or "")))
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
    )

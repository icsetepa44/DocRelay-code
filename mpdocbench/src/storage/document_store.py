"""
Document store — materialises per-page assets under data/documents/<doc>/pages/<page4>/.

MPDocBench ships page images, NO PDFs. The page image IS the Chandra OCR page image
(<ocr root>/<doc>/page_images/page_NNNN.png), so PageData.page_image points straight at it —
no render, no copy. Per page it writes chandra.md, chandra.html, and page_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.core import config
from src.contracts.page import PageData


def _doc_pages_dir(doc_name: str) -> Path:
    return config.DOCUMENTS_DIR / doc_name / "pages"


def resize_image_for_model(image: Image.Image) -> Image.Image:
    """Downsize a copy for the vision model (config.MODEL_IMAGE_MAX_LONG_SIDE)."""
    image = image.convert("RGB")
    long_side_cap = config.MODEL_IMAGE_MAX_LONG_SIDE
    if not long_side_cap:
        return image
    width, height = image.size
    long_side = max(width, height)
    if long_side <= long_side_cap:
        return image
    scale = long_side_cap / long_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        return image.resize(new_size, Image.Resampling.LANCZOS)
    except Exception:
        return image.resize(new_size, Image.LANCZOS)


def ensure_page_assets(record: Dict[str, Any], overwrite: bool = False) -> PageData:
    """Materialise chandra.md/html + page_data.json for one page record; return its PageData.

    The page image is the Chandra OCR page image (``record['page_image_src']``); PageData.page_image
    points straight at it — no PDF render, no copy. ``overwrite`` is accepted for API compatibility.
    """
    doc_name = str(record.get("document_id", ""))
    page_number = int(record.get("page_number", 0))
    content = record.get("content") or {}
    markdown = str(content.get("raw_markdown", "") or "")
    html = str(content.get("raw_html", "") or "")
    layout = dict(record.get("layout") or {})

    page_image_src = str(record.get("page_image_src") or "")
    if not page_image_src or not Path(page_image_src).exists():
        raise FileNotFoundError(
            f"No source page image for {doc_name} page {page_number}: {page_image_src!r}"
        )

    page_dir = _doc_pages_dir(doc_name) / f"{page_number:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "chandra.md").write_text(markdown, encoding="utf-8")
    (page_dir / "chandra.html").write_text(html, encoding="utf-8")

    page_data = PageData(
        document_id=doc_name,
        page_number=page_number,
        page_image=page_image_src,
        markdown=markdown,
        html=html,
        layout=layout,
    )
    (page_dir / "page_data.json").write_text(
        json.dumps(page_data.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return page_data


def load_model_image(page_image_path: str) -> Image.Image:
    """Open a stored page.png and return a model-sized (downsized) copy."""
    return resize_image_for_model(Image.open(page_image_path))


def materialize_document(doc_name: str, overwrite: bool = False) -> List[PageData]:
    """
    Render + persist all page assets for a document (SINGLE-THREADED pre-pass, so
    concurrent model threads never race on writing page.png/chandra.*/page_data.json).
    """
    from src.ocr import chandra_adapter as ocr  # local import avoids any import cycle
    records = ocr.read_page_records(doc_name)
    return [ensure_page_assets(r, overwrite=overwrite) for r in records]


def load_materialized_pages(doc_name: str) -> List[PageData]:
    """Read already-materialised PageData (read-only) from data/documents/<doc>/pages/*/page_data.json."""
    pages_dir = _doc_pages_dir(doc_name)
    out: List[PageData] = []
    for pj in sorted(pages_dir.glob("*/page_data.json")):
        out.append(PageData.from_dict(json.loads(pj.read_text(encoding="utf-8"))))
    out.sort(key=lambda p: p.page_number)
    return out

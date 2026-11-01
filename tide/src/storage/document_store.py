"""
Document store — materialises per-page assets under data/documents/<doc>/pages/<page4>/
and renders the full-page image from the source PDF (closing the "no page image" gap).

Per page it writes:
  page.png         full-page render at config.RENDER_DPI  (0-indexed, 4-digit folder)
  chandra.md       raw Chandra markdown
  chandra.html     raw Chandra html
  page_data.json   the PageData dict

Page-index reconciliation: OCR page_number is 0-indexed; pdf2image is 1-indexed, so the
PDF page rendered is (page_number + 1). Ported from old_pipeline/visual_page_inspector.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from pdf2image import convert_from_path

from src.core import config
from src.contracts.page import PageData


PDF_RENDER_PAGE_OFFSET = 1  # OCR 0-indexed -> pdf2image 1-indexed


def _doc_pages_dir(doc_name: str) -> Path:
    return config.DOCUMENTS_DIR / doc_name / "pages"


def resolve_pdf_path(source_file: str) -> Path:
    """Resolve a page record's source_file (relative to repo root) to an existing PDF."""
    p = Path(str(source_file))
    clean_stem = p.stem.rstrip(".")
    candidates = [
        p,
        config.REPO_ROOT / p,
        Path.cwd() / p,
        config.BENCHMARK_PDF_ROOT / p.name,
        config.BENCHMARK_PDF_ROOT / f"{clean_stem}.pdf",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find source PDF from source_file: {source_file}")


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


def _render_pdf_page(source_file: str, page_number: int, dpi: int) -> Image.Image:
    pdf_path = resolve_pdf_path(source_file)
    pdf_page_idx = page_number  # 0-indexed for pypdfium2

    # 1. Try pypdfium2 first (fast, bundled pdfium, no poppler dependency)
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
        if 0 <= pdf_page_idx < len(pdf):
            page = pdf.get_page(pdf_page_idx)
            # scale factor: 72 points per inch -> scale = dpi / 72
            scale = dpi / 72.0
            pil_image = page.render(scale=scale).to_pil()
            return pil_image
    except Exception:
        pass

    # 2. Fall back to pdf2image
    pdf_page = page_number + PDF_RENDER_PAGE_OFFSET
    try:
        pages = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=pdf_page,
            last_page=pdf_page,
            poppler_path=config.POPPLER_PATH,
        )
        if pages:
            return pages[0]
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF page {pdf_page} for {pdf_path}: {e}")

    raise RuntimeError(f"No image rendered for PDF page {pdf_page}: {pdf_path}")


# Trailing 8 bytes of every complete PNG: the IEND chunk (len=0, type=IEND, fixed CRC).
_PNG_TAIL = b"IEND\xae\x42\x60\x82"


def _png_complete(png_path: Path) -> bool:
    """A fully-written PNG ends with the IEND chunk. A torn write from an interrupted run
    (e.g. a mid-save PC restart) leaves the file present but truncated — it exists yet can't
    be decoded. Such a file must be re-rendered on resume rather than trusted for existing."""
    try:
        with png_path.open("rb") as fh:
            fh.seek(-8, 2)
            return fh.read(8) == _PNG_TAIL
    except OSError:
        return False


def ensure_page_assets(record: Dict[str, Any], overwrite: bool = False) -> PageData:
    """
    Materialise page.png + chandra.md/html + page_data.json for one page record and
    return the corresponding PageData (with page_image pointing at page.png).
    """
    doc_name = str(record.get("document_id", ""))
    page_number = int(record.get("page_number", 0))
    content = record.get("content") or {}
    markdown = str(content.get("raw_markdown", "") or "")
    html = str(content.get("raw_html", "") or "")

    page_dir = _doc_pages_dir(doc_name) / f"{page_number:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    png_path = page_dir / "page.png"

    # Re-render when absent OR when a prior write was torn (truncated) — existence alone
    # is not proof of a valid image; a resume must heal a half-written page.png.
    if overwrite or not _png_complete(png_path):
        image = _render_pdf_page(record.get("source_file", ""), page_number, config.RENDER_DPI)
        image.save(png_path, "PNG")

    (page_dir / "chandra.md").write_text(markdown, encoding="utf-8")
    (page_dir / "chandra.html").write_text(html, encoding="utf-8")

    page_data = PageData(
        document_id=doc_name,
        page_number=page_number,
        page_image=str(png_path),
        markdown=markdown,
        html=html,
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

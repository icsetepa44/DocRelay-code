from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from chandra.model.schema import BatchOutputItem


import sys as _sys
_DU = Path(__file__).resolve().parents[3]
if str(_DU) not in _sys.path:
    _sys.path.insert(0, str(_DU))
from src.core import config

INPUT_MODE: str = "folder"
INPUT_PDF_PATH: str = r"data\input\example.pdf"
INPUT_FOLDER_PATH: str = str(config.BENCHMARK_PDF_ROOT)
GROUP_BY_FOLDER: bool = False
INPUT_LIST: list[str] = [
    str(config.BENCHMARK_PDF_ROOT),
]

OUTPUT_BASE_DIR: str = str(config.BENCHMARK_OCR_PARENT)
FORCE_OVERWRITE: bool = False
PAGE_FILENAME_FMT: str = "page_{:04d}"

LOG_FILE: str = str(config.BENCHMARK_OCR_PARENT / "ocr_chandra.log")
LOG_LEVEL_FILE: str = "DEBUG"
LOG_LEVEL_CONSOLE: str = "INFO"

PAGE_RETRIES: int = 2

OCR_MODE: str = "page"
BATCH_SIZE: int = 4

MODEL_CHECKPOINT: Optional[str] = None
TORCH_DEVICE: Optional[str] = None
TORCH_ATTN: Optional[str] = None

PROMPT_TYPE: str = "ocr"
CUSTOM_PROMPT: Optional[str] = None
MAX_OUTPUT_TOKENS: int = 12384
BBOX_SCALE: int = 1000

INCLUDE_IMAGES: bool = True
INCLUDE_HEADERS_FOOTERS: bool = False

IMAGE_DPI: int = 192
MIN_PDF_IMAGE_DIM: int = 1024
PAGE_RANGE: Optional[list[int]] = None

SAVE_MARKDOWN: bool = True
SAVE_HTML: bool = True
SAVE_LAYOUT: bool = True
SAVE_RAW: bool = True
SAVE_IMAGES: bool = True
SAVE_JSON: bool = True
SAVE_COMBINED: bool = True
SAVE_PAGE_IMAGES: bool = False

TRUNCATION_RATIO: float = 0.98
REOCR_MULTIPLIER: float = 2.0
REOCR_MAX_TOKENS: int = 32768

ENABLE_GRID_CLEAN: bool = False
CLEAN_ON: str = "truncation_or_error"
SAVE_GRID_ARTIFACTS: bool = True
GRID_CONFIG_OVERRIDES: Optional[dict] = None


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, LOG_LEVEL_FILE.upper(), logging.DEBUG))
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL_CONSOLE.upper(), logging.INFO))
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)


@dataclass
class PageResult:
    page_num: int
    out_item: Optional["BatchOutputItem"]
    page_image: Optional["PILImage"]
    truncated: bool
    source: str
    failed: bool = False
    error_msg: Optional[str] = None
    cleaning: Optional[dict] = None
    grid_pipeline: Optional[dict] = None


def is_truncated(token_count: Optional[int], max_tokens: int, ratio: float) -> bool:
    if token_count is None:
        return False
    return token_count >= ratio * max_tokens


def needs_cleaning(out_item: "BatchOutputItem", max_tokens: int) -> bool:
    return bool(getattr(out_item, "error", False)) or is_truncated(
        out_item.token_count, max_tokens, TRUNCATION_RATIO
    )


def reocr_budget() -> int:
    return min(int(MAX_OUTPUT_TOKENS * REOCR_MULTIPLIER), REOCR_MAX_TOKENS)


def num_chunks_of(out_item: "BatchOutputItem") -> int:
    chunks = getattr(out_item, "chunks", None)
    try:
        return len(chunks) if chunks else 0
    except TypeError:
        return 0


def resolve_inputs() -> list[tuple[Path, Path]]:
    base = Path(OUTPUT_BASE_DIR)
    if INPUT_MODE == "single":
        return [_pdf_pair(_check_pdf(Path(INPUT_PDF_PATH)), base)]
    if INPUT_MODE == "folder":
        return _folder_pairs(Path(INPUT_FOLDER_PATH), base)
    if INPUT_MODE == "list":
        if not INPUT_LIST:
            raise ValueError("INPUT_LIST is empty but INPUT_MODE == 'list'")
        pairs: list[tuple[Path, Path]] = []
        for entry in INPUT_LIST:
            p = Path(entry)
            if p.is_dir():
                pairs.extend(_folder_pairs(p, base))
            elif p.suffix.lower() == ".pdf":
                pairs.append(_pdf_pair(_check_pdf(p), base))
            else:
                raise ValueError(
                    f"INPUT_LIST entry is neither a folder nor a .pdf: {p}"
                )
        if not pairs:
            raise FileNotFoundError("INPUT_LIST produced no PDFs to process")
        return pairs
    raise ValueError(
        f"unknown INPUT_MODE {INPUT_MODE!r} (expected 'single', 'folder', or 'list')"
    )


def _check_pdf(pdf: Path) -> Path:
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"not a .pdf: {pdf}")
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    return pdf


def _pdf_pair(pdf: Path, base: Path) -> tuple[Path, Path]:
    return pdf, base / pdf.stem


def _folder_pairs(folder: Path, base: Path) -> list[tuple[Path, Path]]:
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    pdfs = collect_pdf_files(folder)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in folder: {folder}")
    root = base / folder.name if GROUP_BY_FOLDER else base
    return [(pdf, root / pdf.stem) for pdf in pdfs]


def collect_pdf_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf")


def is_already_done(pdf_path: Path, pdf_out_dir: Path) -> bool:
    if SAVE_JSON:
        return (pdf_out_dir / f"{pdf_path.stem}.json").is_file()
    return pdf_out_dir.is_dir() and any(pdf_out_dir.iterdir())


def make_pdf_output_dirs(pdf_out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": pdf_out_dir,
        "markdown": pdf_out_dir / "markdown",
        "html": pdf_out_dir / "html",
        "layout": pdf_out_dir / "layout",
        "images": pdf_out_dir / "images",
        "cleaned": pdf_out_dir / "cleaned",
        "page_images": pdf_out_dir / "page_images",
    }
    dirs["root"].mkdir(parents=True, exist_ok=True)
    if SAVE_MARKDOWN:
        dirs["markdown"].mkdir(exist_ok=True)
    if SAVE_HTML:
        dirs["html"].mkdir(exist_ok=True)
    if SAVE_LAYOUT:
        dirs["layout"].mkdir(exist_ok=True)
    if SAVE_IMAGES:
        dirs["images"].mkdir(exist_ok=True)
    if ENABLE_GRID_CLEAN and SAVE_GRID_ARTIFACTS:
        dirs["cleaned"].mkdir(exist_ok=True)
    if SAVE_PAGE_IMAGES:
        dirs["page_images"].mkdir(exist_ok=True)
    return dirs


_MANAGER = None


def _configure_settings() -> None:
    from chandra.settings import settings

    if MODEL_CHECKPOINT is not None:
        settings.MODEL_CHECKPOINT = MODEL_CHECKPOINT
    if TORCH_DEVICE is not None:
        settings.TORCH_DEVICE = TORCH_DEVICE
    if TORCH_ATTN is not None:
        settings.TORCH_ATTN = TORCH_ATTN
    if BBOX_SCALE is not None:
        settings.BBOX_SCALE = BBOX_SCALE
    if MAX_OUTPUT_TOKENS is not None:
        settings.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS


def get_manager():
    global _MANAGER
    if _MANAGER is None:
        _configure_settings()
        from chandra.model import InferenceManager
        _MANAGER = InferenceManager(method="hf")
    return _MANAGER


def rasterize_pdf(pdf_path: Path) -> list:
    from chandra.input import load_pdf_images

    page_range = PAGE_RANGE or []
    images = load_pdf_images(
        str(pdf_path),
        page_range=page_range,
        image_dpi=IMAGE_DPI,
        min_pdf_image_dim=MIN_PDF_IMAGE_DIM,
    )
    return images


def run_chandra(pil_images: list, max_tokens: int) -> "list[BatchOutputItem]":
    from chandra.model.schema import BatchInputItem

    manager = get_manager()
    prompt_type = None if CUSTOM_PROMPT else PROMPT_TYPE
    batch = [
        BatchInputItem(image=img, prompt=CUSTOM_PROMPT, prompt_type=prompt_type)
        for img in pil_images
    ]

    return manager.generate(
        batch,
        max_output_tokens=max_tokens,
        include_images=INCLUDE_IMAGES,
        include_headers_footers=INCLUDE_HEADERS_FOOTERS,
        bbox_scale=BBOX_SCALE,
    )


def _failed_page(page_num: int, pil_image, exc: Exception) -> PageResult:
    return PageResult(
        page_num=page_num,
        out_item=None,
        page_image=pil_image,
        truncated=False,
        source="original",
        failed=True,
        error_msg=str(exc),
    )


def _ocr_page_with_retry(pil_image, page_num: int, max_tokens: int) -> PageResult:
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(PAGE_RETRIES + 1):
        try:
            out = run_chandra([pil_image], max_tokens)[0]
            return finalize_page(pil_image, out, page_num, max_tokens)
        except Exception as exc:
            last_exc = exc
            if attempt < PAGE_RETRIES:
                logger.warning(
                    "page %04d attempt %d/%d failed: %s — retrying",
                    page_num, attempt + 1, PAGE_RETRIES + 1, exc,
                )
            else:
                logger.error(
                    "page %04d failed all %d attempt(s): %s",
                    page_num, PAGE_RETRIES + 1, exc, exc_info=True,
                )
    return _failed_page(page_num, pil_image, last_exc)


def _ocr_batch_with_retry(batch: list, start: int, max_tokens: int) -> "list[PageResult]":
    end = start + len(batch) - 1
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(PAGE_RETRIES + 1):
        try:
            outs = run_chandra(batch, max_tokens)
            return [finalize_page(batch[j], out, start + j, max_tokens)
                    for j, out in enumerate(outs)]
        except Exception as exc:
            last_exc = exc
            if attempt < PAGE_RETRIES:
                logger.warning(
                    "pages %04d-%04d attempt %d/%d failed: %s — retrying",
                    start, end, attempt + 1, PAGE_RETRIES + 1, exc,
                )
            else:
                logger.error(
                    "pages %04d-%04d failed all %d attempt(s): %s",
                    start, end, PAGE_RETRIES + 1, exc, exc_info=True,
                )
    return [_failed_page(start + j, batch[j], last_exc) for j in range(len(batch))]


def clean_and_reocr(pil_image, first_result: "BatchOutputItem",
                    max_tokens: int) -> "tuple[BatchOutputItem, dict, dict]":
    from src.ocr.generate import grid_clean

    cfg = grid_clean.apply_config_overrides(GRID_CONFIG_OVERRIDES)
    bgr = grid_clean.pil_to_bgr(pil_image)
    pipeline = grid_clean.run_grid_pipeline(bgr, cfg)
    report = pipeline["report"]

    higher = reocr_budget()
    meta = {
        "attempted": True,
        "grid_found": report["regions_found"],
        "grid_lines_removed": report["grid_lines_removed"],
        "dense_grid_region_count": report["dense_grid_region_count"],
        "reocr_max_tokens": higher,
        "original_token_count": first_result.token_count,
        "original_error": bool(getattr(first_result, "error", False)),
        "kept": "original",
    }

    if not report["grid_lines_removed"]:
        return first_result, meta, pipeline

    cleaned_pil = grid_clean.bgr_to_pil(pipeline["cleaned"])
    cleaned_result = run_chandra([cleaned_pil], higher)[0]

    meta["cleaned_token_count"] = cleaned_result.token_count
    meta["cleaned_error"] = bool(getattr(cleaned_result, "error", False))

    if not meta["cleaned_error"]:
        meta["kept"] = "cleaned"
        return cleaned_result, meta, pipeline

    return first_result, meta, pipeline


def finalize_page(pil_image, out_item: "BatchOutputItem", page_num: int,
                  max_tokens: int) -> PageResult:
    truncated = is_truncated(out_item.token_count, max_tokens, TRUNCATION_RATIO)
    source = "original"
    cleaning = None
    grid_pipeline = None

    if ENABLE_GRID_CLEAN and needs_cleaning(out_item, max_tokens):
        out_item, cleaning, grid_pipeline = clean_and_reocr(
            pil_image, out_item, max_tokens
        )
        if cleaning.get("kept") == "cleaned":
            source = "reocr_cleaned"
            truncated = is_truncated(
                out_item.token_count, cleaning["reocr_max_tokens"], TRUNCATION_RATIO
            )

    return PageResult(
        page_num=page_num,
        out_item=out_item,
        page_image=pil_image,
        truncated=truncated,
        source=source,
        cleaning=cleaning,
        grid_pipeline=grid_pipeline,
    )


def save_page_markdown(pr: PageResult, dirs: dict) -> None:
    path = dirs["markdown"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".md")
    path.write_text(pr.out_item.markdown or "", encoding="utf-8")


def save_page_html(pr: PageResult, dirs: dict) -> None:
    path = dirs["html"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".html")
    path.write_text(pr.out_item.html or "", encoding="utf-8")


def save_page_layout(pr: PageResult, dirs: dict) -> None:
    path = dirs["layout"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".json")
    payload = {
        "page_num": pr.page_num,
        "page_box": pr.out_item.page_box,
        "chunks": pr.out_item.chunks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_page_images(pr: PageResult, dirs: dict) -> None:
    images = pr.out_item.images or {}
    prefix = PAGE_FILENAME_FMT.format(pr.page_num)
    for name, img in images.items():
        img.save(str(dirs["images"] / f"{prefix}__{name}"))


def save_page_source_image(pr: PageResult, dirs: dict) -> None:
    path = dirs["page_images"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".png")
    pr.page_image.save(str(path))


def save_grid_artifacts(pr: PageResult, dirs: dict) -> None:
    if not (SAVE_GRID_ARTIFACTS and pr.grid_pipeline is not None):
        return
    from src.ocr.generate import grid_clean

    prefix = PAGE_FILENAME_FMT.format(pr.page_num)
    page_dir = dirs["cleaned"] / prefix
    grid_clean.save_test_artifacts(pr.grid_pipeline, page_dir, image_ref=prefix)


def save_page_artifacts(pr: PageResult, dirs: dict) -> None:
    if pr.failed:
        return
    if SAVE_MARKDOWN:
        save_page_markdown(pr, dirs)
    if SAVE_HTML:
        save_page_html(pr, dirs)
    if SAVE_LAYOUT:
        save_page_layout(pr, dirs)
    if SAVE_IMAGES:
        save_page_images(pr, dirs)
    if SAVE_PAGE_IMAGES:
        save_page_source_image(pr, dirs)
    if ENABLE_GRID_CLEAN:
        save_grid_artifacts(pr, dirs)


def save_combined(results: list[PageResult], pdf_out_dir: Path) -> None:
    if SAVE_MARKDOWN:
        parts = []
        for pr in results:
            md = pr.out_item.markdown if pr.out_item else ""
            parts.append(f"<!-- page {pr.page_num} -->\n\n{md or ''}")
        (pdf_out_dir / "document.md").write_text(
            "\n\n".join(parts), encoding="utf-8"
        )

    if SAVE_HTML:
        parts = []
        for pr in results:
            html = pr.out_item.html if pr.out_item else ""
            parts.append(f"<!-- page {pr.page_num} -->\n{html or ''}")
        (pdf_out_dir / "document.html").write_text(
            "\n<hr/>\n".join(parts), encoding="utf-8"
        )

    if SAVE_LAYOUT:
        payload = [
            {
                "page_num": pr.page_num,
                "page_box": pr.out_item.page_box if pr.out_item else None,
                "chunks": pr.out_item.chunks if pr.out_item else None,
            }
            for pr in results
        ]
        (pdf_out_dir / "document.layout.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def config_snapshot() -> dict:
    return {
        "ocr_mode": OCR_MODE,
        "backend": "hf",
        "model_checkpoint": MODEL_CHECKPOINT,
        "prompt_type": None if CUSTOM_PROMPT else PROMPT_TYPE,
        "custom_prompt": CUSTOM_PROMPT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "bbox_scale": BBOX_SCALE,
        "include_images": INCLUDE_IMAGES,
        "include_headers_footers": INCLUDE_HEADERS_FOOTERS,
        "image_dpi": IMAGE_DPI,
        "min_pdf_image_dim": MIN_PDF_IMAGE_DIM,
        "page_range": PAGE_RANGE,
        "truncation_ratio": TRUNCATION_RATIO,
        "enable_grid_clean": ENABLE_GRID_CLEAN,
        "clean_on": CLEAN_ON,
        "save_grid_artifacts": SAVE_GRID_ARTIFACTS,
        "grid_config_overrides": GRID_CONFIG_OVERRIDES,
        "reocr_multiplier": REOCR_MULTIPLIER,
        "reocr_max_tokens": REOCR_MAX_TOKENS,
    }


def save_json(results: list[PageResult], pdf_path: Path, pdf_out_dir: Path) -> None:
    pages = []
    total_tokens = 0
    total_chunks = 0
    for pr in results:
        out = pr.out_item
        tokens = out.token_count if out else None
        chunks_n = num_chunks_of(out) if out else 0
        total_tokens += tokens or 0
        total_chunks += chunks_n
        page = {
            "page_num": pr.page_num,
            "failed": pr.failed,
            "error_msg": pr.error_msg,
            "page_box": out.page_box if out else None,
            "token_count": tokens,
            "num_chunks": chunks_n,
            "truncated": pr.truncated,
            "error": bool(getattr(out, "error", False)) if out else True,
            "source": pr.source,
            "markdown": out.markdown if out else None,
            "html": out.html if out else None,
            "chunks": out.chunks if out else None,
            "cleaning": pr.cleaning,
        }
        if SAVE_RAW:
            page["raw"] = out.raw if out else None
        pages.append(page)

    doc = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "num_pages": len(results),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "config": config_snapshot(),
        "pages": pages,
    }
    (pdf_out_dir / f"{pdf_path.stem}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ocr_page_by_page(pil_images: list, dirs: dict) -> list[PageResult]:
    results: list[PageResult] = []
    for i, pil in enumerate(pil_images):
        pr = _ocr_page_with_retry(pil, i, MAX_OUTPUT_TOKENS)
        save_page_artifacts(pr, dirs)
        results.append(pr)
        _print_page_badge(pr)
    return results


def ocr_full_document(pil_images: list, dirs: dict) -> list[PageResult]:
    results: list[PageResult] = []
    n = len(pil_images)
    for start in range(0, n, max(1, BATCH_SIZE)):
        batch = pil_images[start:start + max(1, BATCH_SIZE)]
        for pr in _ocr_batch_with_retry(batch, start, MAX_OUTPUT_TOKENS):
            save_page_artifacts(pr, dirs)
            results.append(pr)
            _print_page_badge(pr)

    if SAVE_COMBINED:
        save_combined(results, dirs["root"])
    return results


def _print_page_badge(pr: PageResult) -> None:
    if pr.failed:
        logger.warning("    [fail] page %04d  %s", pr.page_num, pr.error_msg or "")
        return
    out = pr.out_item
    if bool(getattr(out, "error", False)):
        badge = "[err ]"
        log_fn = logger.warning
    elif pr.truncated:
        badge = "[warn]"
        log_fn = logger.warning
    else:
        badge = "[ok  ]"
        log_fn = logger.info
    extra = ""
    if pr.source != "original":
        regions = (pr.cleaning or {}).get("dense_grid_region_count", 0)
        extra = f" (cleaned+reocr, {regions} grid regions)"
    log_fn("    %s page %04d  tokens=%s  chunks=%d%s",
           badge, pr.page_num, out.token_count, num_chunks_of(out), extra)


def process_pdf(pdf_path: Path, pdf_out_dir: Path) -> Optional[list[PageResult]]:
    logger.info("== %s  →  %s", pdf_path.name, pdf_out_dir)

    try:
        pil_images = rasterize_pdf(pdf_path)
    except Exception as exc:
        logger.error("rasterization failed for %s: %s", pdf_path.name, exc, exc_info=True)
        return None

    dirs = make_pdf_output_dirs(pdf_out_dir)
    logger.info("   rasterized %d page(s) at %d DPI; OCR mode=%s",
                len(pil_images), IMAGE_DPI, OCR_MODE)

    if OCR_MODE == "page":
        results = ocr_page_by_page(pil_images, dirs)
    elif OCR_MODE == "full":
        results = ocr_full_document(pil_images, dirs)
    else:
        raise ValueError(f"unknown OCR_MODE {OCR_MODE!r} (expected 'page' or 'full')")

    if SAVE_JSON:
        save_json(results, pdf_path, pdf_out_dir)

    return results


def main() -> None:
    _setup_logging()
    pairs = resolve_inputs()

    logger.info("Chandra PDF OCR — %d PDF(s); output base: %s", len(pairs), OUTPUT_BASE_DIR)
    total_pages = 0
    processed = 0
    skipped = 0
    failed_pdfs = 0
    for pdf_path, pdf_out_dir in pairs:
        if not FORCE_OVERWRITE and is_already_done(pdf_path, pdf_out_dir):
            logger.info("== %s  →  SKIP (already OCR'd: %s)", pdf_path.name, pdf_out_dir)
            skipped += 1
            continue
        try:
            results = process_pdf(pdf_path, pdf_out_dir)
        except Exception as exc:
            logger.error("unexpected error processing %s: %s", pdf_path.name, exc, exc_info=True)
            failed_pdfs += 1
            continue
        if results is None:
            failed_pdfs += 1 
            continue
        failed_pages = sum(1 for r in results if r.failed)
        if failed_pages:
            logger.warning(
                "== %s  completed with %d/%d failed page(s)",
                pdf_path.name, failed_pages, len(results),
            )
        total_pages += len(results)
        processed += 1

    logger.info(
        "DONE — processed %d PDF(s), %d page(s); skipped %d already-done; %d PDF(s) failed.",
        processed, total_pages, skipped, failed_pdfs,
    )


if __name__ == "__main__":
    main()

"""
chandra_ocr.py — Standalone PDF OCR using the Chandra Python API only.

  purpose : OCR one PDF or a whole folder of PDFs with Chandra and save, per PDF,
            per-page Markdown / HTML / layout plus a full-result JSON. Optionally
            recover failed or truncated form pages (empty grid boxes) by removing
            grid lines from the page image and re-OCRing the cleaned image.

  inputs  : a single PDF path, a folder of PDFs, OR a mixed list of folder/PDF
            paths, selected by INPUT_MODE below. Everything is controlled by the
            GLOBAL VARIABLES in the config block; there are no command-line args.
  outputs : per PDF, a directory named after the PDF containing markdown/, html/,
            layout/ folders, an optional cleaned/ folder, and <pdf_stem>.json.
            In full-document mode also combined document.md / document.html /
            document.layout.json for the whole PDF.

Design notes
  * Chandra API only, HuggingFace backend. Uses chandra.input.load_pdf_images
    (pypdfium2) to rasterize PDFs and chandra.model.InferenceManager(method="hf")
    to OCR. No CLI, no subprocess, no PyMuPDF, no vLLM. One InferenceManager is
    loaded once and reused for every page/PDF.
  * InferenceManager.generate() returns a BatchOutputItem per page carrying ALL
    formats at once (.markdown, .html, .chunks, .raw, .page_box, .token_count,
    .images, .error) — so a single call yields markdown + html + layout.
  * prompt_type is pinned to "ocr_layout" (layout-preserving mode). Chandra is not
    instruction-following; CUSTOM_PROMPT is exposed but normally left None.
  * Grid cleaning is delegated to grid_clean.py (a thin reuse of grid_detection.py). It runs
    ONLY when a page's OCR failed or was truncated, then re-OCRs the cleaned image
    and keeps the better result.
  * Heavy imports (chandra, PIL, cv2, grid_clean) are lazy so the module imports
    cleanly for inspection without loading a model.
  * Outputs are never silently overwritten: when FORCE_OVERWRITE is False, any PDF
    that is already fully OCR'd (its completion marker is present) is skipped;
    FORCE_OVERWRITE=True reprocesses everything.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Type-checker only: gives Pylance the real Chandra/PIL types without paying
    # the heavy import cost at runtime (these modules are imported lazily below).
    from PIL.Image import Image as PILImage
    from chandra.model.schema import BatchOutputItem

# ===========================================================================
# GLOBAL CONFIG — the only control surface for this script (no CLI args)
# ===========================================================================

# --- Make `src.*` importable whether run via run.py or standalone -----------
import sys as _sys
_DU = Path(__file__).resolve().parents[3]  # tide/
if str(_DU) not in _sys.path:
    _sys.path.insert(0, str(_DU))
from src.core import config  # noqa: E402  (OCR I/O paths centralized in config)

# --- Input -----------------------------------------------------------------
INPUT_MODE: str = "folder"                       # "single" (one PDF) | "folder" (all PDFs in a dir) | "list" (mixed)
INPUT_PDF_PATH: str = r"data\input\example.pdf"  # used when INPUT_MODE == "single"
INPUT_FOLDER_PATH: str = str(config.BENCHMARK_PDF_ROOT)  # drop the PDFs you want to test here (OCR input)
GROUP_BY_FOLDER: bool = False                    # False -> OUTPUT_BASE_DIR/<stem>/  (flat, our layout);
                                                 # True  -> OUTPUT_BASE_DIR/<folder_name>/<stem>/
INPUT_LIST: list[str] = [                        # used only when INPUT_MODE == "list"; mix folders and/or .pdf files
    str(config.BENCHMARK_PDF_ROOT),
]

# --- Output ----------------------------------------------------------------
OUTPUT_BASE_DIR: str = str(config.BENCHMARK_OCR_PARENT)  # pipeline input lives here (chandra structure)
FORCE_OVERWRITE: bool = False                    # True | False — True lets a re-run overwrite a non-empty output dir
PAGE_FILENAME_FMT: str = "page_{:04d}"           # str.format template, 0-indexed → page_0000, page_0001, ...

# --- Logging ---------------------------------------------------------------
LOG_FILE: str = str(config.BENCHMARK_OCR_PARENT / "ocr_chandra.log")  # shared log file (appended each run)
LOG_LEVEL_FILE: str = "DEBUG"                    # file handler level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_LEVEL_CONSOLE: str = "INFO"                  # console handler level: same options

# --- Resilience ------------------------------------------------------------
PAGE_RETRIES: int = 2                            # int ≥ 0 — extra OCR attempts per page/batch before marking failed

# --- OCR mode --------------------------------------------------------------
OCR_MODE: str = "page"                           # "page" (one generate() per page) | "full" (batched + combined files)
BATCH_SIZE: int = 4                              # int ≥ 1 — pages per generate() call in "full" mode (VRAM ↔ speed)

# --- Chandra model (HuggingFace backend only) ------------------------------
MODEL_CHECKPOINT: Optional[str] = None           # None (→ "datalab-to/chandra-ocr-2") | any HF model id / local path
TORCH_DEVICE: Optional[str] = None               # None (auto) | "cuda" | "cuda:0" | "cpu"
TORCH_ATTN: Optional[str] = None                 # None (default) | "flash_attention_2" | "sdpa" | "eager"

# --- OCR options -----------------------------------------------------------
PROMPT_TYPE: str = "ocr"                  # "ocr_layout" (layout + bbox) | "ocr" (plain) — only these two exist
CUSTOM_PROMPT: Optional[str] = None              # None | any prompt str (overrides PROMPT_TYPE; normally leave None)
MAX_OUTPUT_TOKENS: int = 12384                   # int > 0 — generation cap per page (first pass)
BBOX_SCALE: int = 1000                           # int > 0 — layout bbox normalization scale (Chandra default 1000)

# --- Layout / rendering options -------------------------------------------
INCLUDE_IMAGES: bool = True                      # True | False — keep image refs / extract cropped images
INCLUDE_HEADERS_FOOTERS: bool = False            # True | False — keep page headers/footers in md/html

# --- PDF rasterization -----------------------------------------------------
IMAGE_DPI: int = 192                             # int > 0 — DPI for PDF → image rendering (pypdfium2)
MIN_PDF_IMAGE_DIM: int = 1024                    # int > 0 — min page-image dimension (auto up-scale below this)
PAGE_RANGE: Optional[list[int]] = None           # None (all pages) | 0-indexed list, e.g. [0, 1, 2]

# --- Output formats (what to save) ----------------------------------------
SAVE_MARKDOWN: bool = True                        # True | False — markdown/page_NNNN.md
SAVE_HTML: bool = True                            # True | False — html/page_NNNN.html
SAVE_LAYOUT: bool = True                          # True | False — layout/page_NNNN.json (chunks)
SAVE_RAW: bool = True                             # True | False — include raw model HTML in the JSON
SAVE_IMAGES: bool = True                          # True | False — dump extracted images to images/
SAVE_JSON: bool = True                            # True | False — <pdf_stem>.json (full result + metadata)
SAVE_COMBINED: bool = True                        # True | False — combined document.* files (full-doc mode only)
SAVE_PAGE_IMAGES: bool = False                    # True | False — save the rasterized page PNGs to page_images/

# --- Truncation recovery ---------------------------------------------------
TRUNCATION_RATIO: float = 0.98                    # 0.0–1.0 — token_count ≥ ratio × cap ⇒ page counted as truncated
REOCR_MULTIPLIER: float = 2.0                     # float > 1 — cleaned re-OCR budget = cap × this ...
REOCR_MAX_TOKENS: int = 32768                     # int > 0 — ... clamped to this ceiling

# --- Grid cleaning (post-processing of failed/truncated form pages) -------
ENABLE_GRID_CLEAN: bool = False                   # True | False — master switch for the clean-then-reocr path
CLEAN_ON: str = "truncation_or_error"             # "truncation_or_error" — trigger condition (currently the only mode)
SAVE_GRID_ARTIFACTS: bool = True                  # True | False — write the full grid_detection.py debug set per cleaned page
GRID_CONFIG_OVERRIDES: Optional[dict] = None       # None | dict of grid_detection.GridDetectionConfig fields,
                                                   #e.g. {"min_vertical_lines": 6, "adaptive_c": 12}


# ===========================================================================
# LOGGING
# ===========================================================================
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure the root logger with a file handler and a console handler.

    Called once from main(). The log file is appended (not overwritten) so
    multiple runs accumulate in one place. Both handlers share the same
    formatter; only their levels differ.
    """
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


# ===========================================================================
# Working transport (plain dataclass carrying a page's OCR result + metadata)
# ===========================================================================
@dataclass
class PageResult:
    """One page's OCR outcome plus the metadata the saving/JSON steps need."""
    page_num: int
    out_item: Optional["BatchOutputItem"]  # None when all retries failed
    page_image: Optional["PILImage"]       # source PIL image (for optional page-image save)
    truncated: bool
    source: str                            # "original" | "reocr_cleaned"
    failed: bool = False                   # True when all OCR retries were exhausted
    error_msg: Optional[str] = None        # exception message on failure, else None
    cleaning: Optional[dict] = None        # cleaning metadata, or None if not attempted
    grid_pipeline: Optional[dict] = None   # grid_clean.run_grid_pipeline bundle, or None


# ===========================================================================
# PURE LOGIC
# ===========================================================================
def is_truncated(token_count: Optional[int], max_tokens: int, ratio: float) -> bool:
    """True if generation hit (near) the cap, i.e. the page was cut off.

    token_count : tokens generated for the page (None ⇒ unknown)
    max_tokens  : the cap used for that generation
    ratio       : fraction of the cap that counts as "truncated"
    Returns     : bool
    """
    if token_count is None:
        return False
    return token_count >= ratio * max_tokens


def needs_cleaning(out_item: "BatchOutputItem", max_tokens: int) -> bool:
    """True if a page's OCR result warrants a grid-clean + re-OCR attempt.

    out_item   : a chandra BatchOutputItem
    max_tokens : the cap used for that page
    Returns    : bool (errored OR truncated)
    """
    return bool(getattr(out_item, "error", False)) or is_truncated(
        out_item.token_count, max_tokens, TRUNCATION_RATIO
    )


def reocr_budget() -> int:
    """Compute the token budget for a cleaned re-OCR pass.

    Returns : min(MAX_OUTPUT_TOKENS × REOCR_MULTIPLIER, REOCR_MAX_TOKENS) as int.
    """
    return min(int(MAX_OUTPUT_TOKENS * REOCR_MULTIPLIER), REOCR_MAX_TOKENS)


def num_chunks_of(out_item: "BatchOutputItem") -> int:
    """Number of layout chunks in a BatchOutputItem (0 when absent)."""
    chunks = getattr(out_item, "chunks", None)
    try:
        return len(chunks) if chunks else 0
    except TypeError:
        return 0


# ===========================================================================
# INPUT RESOLUTION
# ===========================================================================
def resolve_inputs() -> list[tuple[Path, Path]]:
    """Decide the (PDF, per-PDF output directory) pairs from the input globals.

    Returns : list of (pdf_path, pdf_out_dir) where pdf_out_dir is the directory
              that will hold that PDF's output tree. Output layout by source:
                single mode / a listed PDF  → OUTPUT_BASE_DIR/<pdf_stem>
                folder mode / a listed dir  → OUTPUT_BASE_DIR/<folder_name>/<pdf_stem>
              (Grouping listed folders under their own name keeps PDFs with the
              same stem from different folders out of each other's way.)
    """
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
    """Validate that `pdf` is an existing .pdf file; return it unchanged."""
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"not a .pdf: {pdf}")
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    return pdf


def _pdf_pair(pdf: Path, base: Path) -> tuple[Path, Path]:
    """Pair a directly-listed PDF with its output dir (base/<stem>)."""
    return pdf, base / pdf.stem


def _folder_pairs(folder: Path, base: Path) -> list[tuple[Path, Path]]:
    """Expand one folder into (pdf, base/<folder_name>/<stem>) pairs."""
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    pdfs = collect_pdf_files(folder)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in folder: {folder}")
    root = base / folder.name if GROUP_BY_FOLDER else base
    return [(pdf, root / pdf.stem) for pdf in pdfs]


def collect_pdf_files(folder: Path) -> list[Path]:
    """Collect PDF files directly inside a folder, sorted by name.

    folder  : directory to scan (non-recursive)
    Returns : sorted list of Path to *.pdf files
    """
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf")


def is_already_done(pdf_path: Path, pdf_out_dir: Path) -> bool:
    """True if this PDF appears to have been fully OCR'd already.

    Used to skip completed work when FORCE_OVERWRITE is False. The completion
    marker is the final <pdf_stem>.json (written last by process_pdf); when JSON
    output is disabled, fall back to "the output dir already holds files".

    pdf_path    : the source PDF
    pdf_out_dir : that PDF's output directory
    Returns     : bool
    """
    if SAVE_JSON:
        return (pdf_out_dir / f"{pdf_path.stem}.json").is_file()
    return pdf_out_dir.is_dir() and any(pdf_out_dir.iterdir())


# ===========================================================================
# OUTPUT DIRECTORY LAYOUT
# ===========================================================================
def make_pdf_output_dirs(pdf_out_dir: Path) -> dict[str, Path]:
    """Create the output folder tree for one PDF; refuse silent overwrite.

    pdf_out_dir : the per-PDF output directory (named after the PDF stem)
    Returns     : dict of subdir paths keyed "root","markdown","html","layout",
                  "images","cleaned","page_images".
    Saves       : creates directories on disk.
    Note        : overwrite protection lives in main() — when FORCE_OVERWRITE is
                  False, PDFs already OCR'd are skipped before we get here, so a
                  dir reaching this point is either fresh or a forced/partial redo.
    """
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


# ===========================================================================
# CHANDRA MODEL (loaded once, reused for all pages/PDFs)
# ===========================================================================
_MANAGER = None  # singleton InferenceManager


def _configure_settings() -> None:
    """Push the relevant model globals onto Chandra's settings singleton.

    Called once before the InferenceManager is built. Only overrides settings for
    globals that are set (non-None), leaving Chandra defaults otherwise.
    """
    from chandra.settings import settings  # lazy

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
    """Load (and cache) the Chandra HuggingFace InferenceManager.

    Returns : a chandra.model.InferenceManager (loaded once per process).
    """
    global _MANAGER
    if _MANAGER is None:
        _configure_settings()
        from chandra.model import InferenceManager  # lazy — heavy import
        _MANAGER = InferenceManager(method="hf")
    return _MANAGER


# ===========================================================================
# RASTERIZE + OCR
# ===========================================================================
def rasterize_pdf(pdf_path: Path) -> list:
    """Render a PDF to per-page PIL images via the Chandra PDF loader.

    pdf_path : path to the input PDF
    Returns  : list of PIL.Image (one per selected page), RGB.
    """
    from chandra.input import load_pdf_images  # lazy

    # load_pdf_images treats a falsy page_range as "all pages" (see chandra.input:
    # `if not page_range or page in page_range`). Pass [] rather than None so the
    # call matches the annotated List[int] type while keeping the same behavior.
    page_range = PAGE_RANGE or []
    images = load_pdf_images(
        str(pdf_path),
        page_range=page_range,
        image_dpi=IMAGE_DPI,
        min_pdf_image_dim=MIN_PDF_IMAGE_DIM,
    )
    return images


def run_chandra(pil_images: list, max_tokens: int) -> "list[BatchOutputItem]":
    """Run Chandra OCR on a batch of page images.

    pil_images : list of PIL images to OCR together
    max_tokens : generation cap for this call
    Returns    : list of chandra BatchOutputItem (one per input image), each with
                 .markdown/.html/.chunks/.raw/.page_box/.token_count/.images/.error
    """
    from chandra.model.schema import BatchInputItem  # lazy

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


# ===========================================================================
# RETRY WRAPPERS
# ===========================================================================
def _failed_page(page_num: int, pil_image, exc: Exception) -> PageResult:
    """Build a stub PageResult marking a page that exhausted all retries."""
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
    """OCR a single page with up to PAGE_RETRIES extra attempts on exception.

    Returns a failed PageResult (out_item=None, failed=True) rather than
    raising, so the calling loop can continue with the next page.
    """
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
    """OCR a batch of pages with up to PAGE_RETRIES extra attempts on exception.

    Returns failed PageResults for every page in the batch on total failure.
    """
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


# ===========================================================================
# GRID CLEANING + RE-OCR (only on failed/truncated pages)
# ===========================================================================
def clean_and_reocr(pil_image, first_result: "BatchOutputItem",
                    max_tokens: int) -> "tuple[BatchOutputItem, dict, dict]":
    """Remove grid lines from a bad page's image and re-OCR the cleaned image.

    pil_image    : the source page PIL image
    first_result : the (bad) first-pass BatchOutputItem
    max_tokens   : the cap used for the first pass
    Returns      : (best_result, cleaning_meta, pipeline) where
                     best_result  is the kept BatchOutputItem (cleaned or original)
                     cleaning_meta is a JSON-serializable dict describing the attempt
                     pipeline      is grid_clean.run_grid_pipeline's bundle (masks,
                                   regions, cleaned image, report) for saving
    Saves        : nothing (caller persists artifacts).
    """
    from src.ocr.generate import grid_clean  # lazy — pulls in cv2

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

    # If no dense grid was found, cleaning cannot help — keep the original result.
    if not report["grid_lines_removed"]:
        return first_result, meta, pipeline

    cleaned_pil = grid_clean.bgr_to_pil(pipeline["cleaned"])
    cleaned_result = run_chandra([cleaned_pil], higher)[0]

    meta["cleaned_token_count"] = cleaned_result.token_count
    meta["cleaned_error"] = bool(getattr(cleaned_result, "error", False))

    # The first pass was already known-bad (that is why we are here). Keep the
    # cleaned result unless it itself errored.
    if not meta["cleaned_error"]:
        meta["kept"] = "cleaned"
        return cleaned_result, meta, pipeline

    return first_result, meta, pipeline


def finalize_page(pil_image, out_item: "BatchOutputItem", page_num: int,
                  max_tokens: int) -> PageResult:
    """Wrap a first-pass OCR result, optionally cleaning+re-OCRing bad pages.

    pil_image  : the source page PIL image
    out_item   : the first-pass BatchOutputItem for this page
    page_num   : 0-indexed page number
    max_tokens : the cap used for the first pass
    Returns    : a PageResult with the kept result and its metadata.
    """
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


# ===========================================================================
# SAVE — per page
# ===========================================================================
def save_page_markdown(pr: PageResult, dirs: dict) -> None:
    """Save a page's Markdown to markdown/page_NNNN.md."""
    path = dirs["markdown"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".md")
    path.write_text(pr.out_item.markdown or "", encoding="utf-8")


def save_page_html(pr: PageResult, dirs: dict) -> None:
    """Save a page's cleaned HTML to html/page_NNNN.html."""
    path = dirs["html"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".html")
    path.write_text(pr.out_item.html or "", encoding="utf-8")


def save_page_layout(pr: PageResult, dirs: dict) -> None:
    """Save a page's layout chunks (bbox+label+content) to layout/page_NNNN.json."""
    path = dirs["layout"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".json")
    payload = {
        "page_num": pr.page_num,
        "page_box": pr.out_item.page_box,
        "chunks": pr.out_item.chunks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_page_images(pr: PageResult, dirs: dict) -> None:
    """Save Chandra's extracted per-page images to images/ (prefixed by page)."""
    images = pr.out_item.images or {}
    prefix = PAGE_FILENAME_FMT.format(pr.page_num)
    for name, img in images.items():
        img.save(str(dirs["images"] / f"{prefix}__{name}"))


def save_page_source_image(pr: PageResult, dirs: dict) -> None:
    """Save the rasterized source page image to page_images/page_NNNN.png."""
    path = dirs["page_images"] / (PAGE_FILENAME_FMT.format(pr.page_num) + ".png")
    pr.page_image.save(str(path))


def save_grid_artifacts(pr: PageResult, dirs: dict) -> None:
    """Save the full grid_detection.py debug artifact set for a page the cleaner ran on.

    Writes the 8 files grid_detection.py.process_image produces into
    cleaned/page_NNNN/ (one subfolder per page). No-op unless the grid pipeline
    actually ran for this page (i.e. cleaning was attempted).

    pr   : the PageResult (must carry a grid_pipeline bundle)
    dirs : the output directory map from make_pdf_output_dirs()
    Saves: cleaned/page_NNNN/{dense_grid_regions.json, *_marked.png, *_debug.png,
           cleaned_dense_grids_only.png, removed_dense_grid_line_mask.png,
           raw_line_mask.png, horizontal_line_mask.png, vertical_line_mask.png}
    """
    if not (SAVE_GRID_ARTIFACTS and pr.grid_pipeline is not None):
        return
    from src.ocr.generate import grid_clean  # lazy — pulls in cv2

    prefix = PAGE_FILENAME_FMT.format(pr.page_num)
    page_dir = dirs["cleaned"] / prefix
    grid_clean.save_test_artifacts(pr.grid_pipeline, page_dir, image_ref=prefix)


def save_page_artifacts(pr: PageResult, dirs: dict) -> None:
    """Save all enabled per-page artifacts for one page.

    pr   : the PageResult to persist
    dirs : the output directory map from make_pdf_output_dirs()
    Saves: markdown/html/layout/images/cleaned/page-image files per the SAVE_* flags.
    No-op for failed pages (out_item is None); the failure is recorded in the JSON.
    """
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


# ===========================================================================
# SAVE — combined (full-document mode) and JSON
# ===========================================================================
def save_combined(results: list[PageResult], pdf_out_dir: Path) -> None:
    """Write combined whole-PDF Markdown/HTML/layout files.

    results     : all PageResults for the PDF, in page order
    pdf_out_dir : the per-PDF output directory
    Saves       : document.md, document.html, document.layout.json (per SAVE_* flags).
    """
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
    """Return a JSON-serializable snapshot of the OCR-relevant config globals."""
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
    """Write <pdf_stem>.json holding the full OCR result and metadata.

    results     : all PageResults for the PDF, in page order
    pdf_path    : the source PDF path
    pdf_out_dir : the per-PDF output directory
    Saves       : pdf_out_dir/<pdf_stem>.json
    """
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


# ===========================================================================
# OCR DRIVERS (page-by-page vs full-document)
# ===========================================================================
def ocr_page_by_page(pil_images: list, dirs: dict) -> list[PageResult]:
    """OCR one page at a time, saving each page's artifacts incrementally.

    pil_images : rasterized page images in order
    dirs       : output directory map for this PDF
    Returns    : list of PageResult (page order)
    Saves      : per-page artifacts as each page completes.
    """
    results: list[PageResult] = []
    for i, pil in enumerate(pil_images):
        pr = _ocr_page_with_retry(pil, i, MAX_OUTPUT_TOKENS)
        save_page_artifacts(pr, dirs)
        results.append(pr)
        _print_page_badge(pr)
    return results


def ocr_full_document(pil_images: list, dirs: dict) -> list[PageResult]:
    """OCR the whole PDF in batches, then also write combined document files.

    pil_images : rasterized page images in order
    dirs       : output directory map for this PDF
    Returns    : list of PageResult (page order)
    Saves      : per-page artifacts + combined document.* files (if SAVE_COMBINED).
    """
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


# ===========================================================================
# PER-PDF ORCHESTRATION + progress
# ===========================================================================
def _print_page_badge(pr: PageResult) -> None:
    """Log a one-line status badge for a finished page."""
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
    """Rasterize + OCR one PDF and save all outputs.

    pdf_path    : the input PDF
    pdf_out_dir : the per-PDF output directory (named after the PDF stem)
    Returns     : list of PageResult, or None when the PDF could not be rasterized
                  (None signals a hard failure so main() can count it correctly).
    Saves       : the full per-PDF output tree (per-page files, optional combined
                  files, and <pdf_stem>.json).
    """
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


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    """Resolve inputs and OCR every selected PDF into its own output directory.

    When FORCE_OVERWRITE is False, PDFs that are already fully OCR'd are skipped.
    A PDF that fails entirely (rasterization error or unexpected exception) is
    logged and skipped; the run continues with remaining PDFs.
    Saves : one output directory per PDF under OUTPUT_BASE_DIR.
    """
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

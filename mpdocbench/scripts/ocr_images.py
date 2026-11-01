from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chandra.model.schema import BatchOutputItem

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

INPUT_IMAGES_DIR: Path = REPO_ROOT / "data" / "mpdocbench" / "images"
OUTPUT_DIR: Path = REPO_ROOT / "data" / "mpdocbench" / "ocr"
IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png")

MODEL_CHECKPOINT: str = os.environ.get("CHANDRA_MODEL", "datalab-to/chandra-ocr-2")
TORCH_DEVICE: Optional[str] = os.environ.get("CHANDRA_DEVICE") or None
TORCH_ATTN: Optional[str] = os.environ.get("CHANDRA_ATTN") or None

PROMPT_TYPE: str = "ocr_layout"
MAX_OUTPUT_TOKENS: int = 12384
BBOX_SCALE: int = 1000
INCLUDE_IMAGES: bool = True
INCLUDE_HEADERS_FOOTERS: bool = True
MIN_IMAGE_DIM: int = 1536

PAGE_RETRIES: int = 2
RESUMABLE: bool = True
FORCE_OVERWRITE: bool = False

PAGE_FILENAME_FMT: str = "page_{:04d}"
SAVE_MARKDOWN: bool = True
SAVE_HTML: bool = True
SAVE_LAYOUT: bool = True
SAVE_IMAGES: bool = True
SAVE_PAGE_IMAGES: bool = True
SAVE_RAW: bool = True

TRUNCATION_RATIO: float = 0.98

LOG_LEVEL_CONSOLE: str = "INFO"
LOG_LEVEL_FILE: str = "DEBUG"


logger = logging.getLogger("mpdocbench_ocr")


def _setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(OUTPUT_DIR / "run_ocr.log", encoding="utf-8")
    fh.setLevel(getattr(logging, LOG_LEVEL_FILE.upper(), logging.DEBUG))
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL_CONSOLE.upper(), logging.INFO))
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)


_MANAGER = None


def get_manager():
    global _MANAGER
    if _MANAGER is None:
        from chandra.settings import settings
        settings.MODEL_CHECKPOINT = MODEL_CHECKPOINT
        if TORCH_DEVICE is not None:
            settings.TORCH_DEVICE = TORCH_DEVICE
        if TORCH_ATTN is not None:
            settings.TORCH_ATTN = TORCH_ATTN
        settings.BBOX_SCALE = BBOX_SCALE
        settings.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
        from chandra.model import InferenceManager
        logger.info("Loading Chandra InferenceManager (hf) — model=%s", MODEL_CHECKPOINT)
        _MANAGER = InferenceManager(method="hf")
    return _MANAGER


def _page_int(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def num_chunks_of(out_item) -> int:
    chunks = getattr(out_item, "chunks", None)
    try:
        return len(chunks) if chunks else 0
    except TypeError:
        return 0


def is_truncated(token_count: Optional[int], max_tokens: int, ratio: float) -> bool:
    if token_count is None:
        return False
    return token_count >= ratio * max_tokens


def list_doc_folders(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"INPUT_IMAGES_DIR not found: {root}")
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()):
            out.append(p)
    return out


def list_page_images(doc_folder: Path) -> list[Path]:
    imgs = [f for f in doc_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS and f.stem.startswith("page_")]
    return sorted(imgs, key=_page_int)


def make_output_dirs(doc_out_dir: Path) -> dict:
    dirs = {
        "root": doc_out_dir,
        "markdown": doc_out_dir / "markdown",
        "html": doc_out_dir / "html",
        "layout": doc_out_dir / "layout",
        "images": doc_out_dir / "images",
        "page_images": doc_out_dir / "page_images",
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
    if SAVE_PAGE_IMAGES:
        dirs["page_images"].mkdir(exist_ok=True)
    return dirs


def save_page_artifacts(page_num: int, out_item, page_image, dirs: dict) -> None:
    prefix = PAGE_FILENAME_FMT.format(page_num)
    if SAVE_MARKDOWN:
        (dirs["markdown"] / f"{prefix}.md").write_text(out_item.markdown or "", encoding="utf-8")
    if SAVE_HTML:
        (dirs["html"] / f"{prefix}.html").write_text(out_item.html or "", encoding="utf-8")
    if SAVE_LAYOUT:
        payload = {
            "page_num": page_num,
            "page_box": out_item.page_box,
            "chunks": out_item.chunks,
        }
        (dirs["layout"] / f"{prefix}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if SAVE_IMAGES:
        for name, img in (out_item.images or {}).items():
            img.save(str(dirs["images"] / f"{prefix}__{name}"))
    if SAVE_PAGE_IMAGES and page_image is not None:
        page_image.save(str(dirs["page_images"] / f"{prefix}.png"))


def config_snapshot() -> dict:
    return {
        "benchmark": "mpdocbench",
        "backend": "hf",
        "model_checkpoint": MODEL_CHECKPOINT,
        "prompt_type": PROMPT_TYPE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "bbox_scale": BBOX_SCALE,
        "include_images": INCLUDE_IMAGES,
        "include_headers_footers": INCLUDE_HEADERS_FOOTERS,
        "min_image_dim": MIN_IMAGE_DIM,
        "source": "mpdocbench_images",
    }


def save_marker(doc_name: str, doc_folder: Path, pages: list[dict], doc_out_dir: Path) -> None:
    total_tokens = sum((p.get("token_count") or 0) for p in pages)
    total_chunks = sum((p.get("num_chunks") or 0) for p in pages)
    doc = {
        "file_name": f"{doc_name}.pdf",
        "file_path": str(doc_folder),
        "num_pages": len(pages),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "config": config_snapshot(),
        "pages": pages,
    }
    (doc_out_dir / f"{doc_name}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def ocr_image(manager, pil_image):
    from chandra.model.schema import BatchInputItem
    last_err = None
    for attempt in range(PAGE_RETRIES + 1):
        try:
            out_items = manager.generate(
                [BatchInputItem(image=pil_image, prompt=None, prompt_type=PROMPT_TYPE)],
                max_output_tokens=MAX_OUTPUT_TOKENS,
                include_images=INCLUDE_IMAGES,
                include_headers_footers=INCLUDE_HEADERS_FOOTERS,
                bbox_scale=BBOX_SCALE,
            )
            return out_items[0], None
        except Exception as exc:
            last_err = str(exc)
            logger.warning("OCR attempt %d/%d failed: %s", attempt + 1, PAGE_RETRIES + 1, last_err)
    return None, last_err


def process_doc(doc_folder: Path) -> None:
    from chandra.input import load_image

    doc_name = doc_folder.name
    doc_out_dir = OUTPUT_DIR / doc_name
    marker = doc_out_dir / f"{doc_name}.json"

    if RESUMABLE and not FORCE_OVERWRITE and marker.is_file():
        logger.info("SKIP %s (marker present)", doc_name)
        return

    image_paths = list_page_images(doc_folder)
    if not image_paths:
        logger.warning("SKIP %s (no page images)", doc_name)
        return

    dirs = make_output_dirs(doc_out_dir)
    manager = get_manager()
    logger.info("DOC %s — %d page(s)", doc_name, len(image_paths))

    pages: list[dict] = []
    for idx, img_path in enumerate(image_paths):
        pil_image = load_image(str(img_path), min_image_dim=MIN_IMAGE_DIM)
        out_item, err = ocr_image(manager, pil_image)

        if out_item is None:
            logger.error("  page %04d (%s) FAILED: %s", idx, img_path.name, err)
            pages.append({
                "page_num": idx,
                "failed": True,
                "error_msg": err,
                "page_box": None,
                "token_count": None,
                "num_chunks": 0,
                "truncated": False,
                "error": True,
                "source": "original",
                "source_image": img_path.name,
                "markdown": None,
                "html": None,
                "chunks": None,
                "cleaning": None,
                **({"raw": None} if SAVE_RAW else {}),
            })
            continue

        save_page_artifacts(idx, out_item, pil_image, dirs)
        chunks_n = num_chunks_of(out_item)
        truncated = is_truncated(out_item.token_count, MAX_OUTPUT_TOKENS, TRUNCATION_RATIO)
        logger.info("  page %04d (%s) ok — %d chunks, %s tokens%s",
                    idx, img_path.name, chunks_n, out_item.token_count,
                    " [truncated]" if truncated else "")
        page = {
            "page_num": idx,
            "failed": False,
            "error_msg": None,
            "page_box": out_item.page_box,
            "token_count": out_item.token_count,
            "num_chunks": chunks_n,
            "truncated": truncated,
            "error": bool(getattr(out_item, "error", False)),
            "source": "original",
            "source_image": img_path.name,
            "markdown": out_item.markdown,
            "html": out_item.html,
            "chunks": out_item.chunks,
            "cleaning": None,
        }
        if SAVE_RAW:
            page["raw"] = out_item.raw
        pages.append(page)

    save_marker(doc_name, doc_folder, pages, doc_out_dir)
    logger.info("DONE %s -> %s", doc_name, doc_out_dir)


def main() -> None:
    _setup_logging()
    logger.info("MPDocBench OCR — input=%s  output=%s", INPUT_IMAGES_DIR, OUTPUT_DIR)
    doc_folders = list_doc_folders(INPUT_IMAGES_DIR)
    logger.info("Found %d document folder(s)", len(doc_folders))
    done = 0
    for i, doc_folder in enumerate(doc_folders, 1):
        logger.info("[%d/%d] %s", i, len(doc_folders), doc_folder.name)
        try:
            process_doc(doc_folder)
            done += 1
        except Exception as exc:
            logger.exception("Document %s crashed: %s", doc_folder.name, exc)
    logger.info("All done. Processed %d/%d document folder(s).", done, len(doc_folders))


if __name__ == "__main__":
    main()

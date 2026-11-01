"""
grid_clean.py — Pre-OCR image cleaning helper (dense form-grid line removal).

  purpose : reusable wrapper around the grid-detection/removal logic in grid_detection.py so
            an OCR script can clean a page image in memory (remove empty grid-box
            ruled lines) and re-OCR it, without duplicating any grid math. Also
            reproduces the full set of debug artifacts grid_detection.py saves per page.
  stage   : helper for Stage 1 (perception) — runs BETWEEN a bad OCR attempt and a
            re-OCR of the cleaned image. Not a standalone pipeline stage.
  inputs  : an in-memory page image (PIL image or OpenCV BGR numpy array) and an
            optional GridDetectionConfig.
  outputs : a cleaned OpenCV BGR numpy array + a grid-pipeline bundle (masks,
            regions, JSON-serializable report). save_test_artifacts() persists the
            same 8 files grid_detection.py.process_image() writes.

Design notes
  * All detection/removal/drawing/JSON is imported from grid_detection.py (single source of
    truth): detect_dense_grid_regions(), remove_dense_grid_lines(), draw_regions(),
    draw_dense_and_rejected_regions(), save_json(). This file only adds thin
    PIL<->cv2 converters, an in-memory pipeline runner, and an artifact saver.
  * Importing grid_detection.py does NOT run its process_image() — that lives under its own
    `if __name__ == "__main__":` guard.
  * Heavy imports (cv2, numpy) are module-level: this helper is only ever used on a
    machine that already has OpenCV, so there is no lazy-import benefit.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Make `src.*` importable whether run via run.py or standalone.
import sys as _sys
_DU = Path(__file__).resolve().parents[3]  # tide/
if str(_DU) not in _sys.path:
    _sys.path.insert(0, str(_DU))

# Single source of truth for the grid math — reuse, do not re-implement.
from src.ocr.generate.grid_detection import (
    CONFIG,
    GridDetectionConfig,
    detect_dense_grid_regions,
    draw_dense_and_rejected_regions,
    draw_regions,
    remove_dense_grid_lines,
    save_json as test_save_json,
)


# ---------------------------------------------------------------------------
# PIL <-> OpenCV converters
# ---------------------------------------------------------------------------
def pil_to_bgr(pil_image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR numpy array.

    pil_image : a PIL.Image.Image (any mode; forced to RGB first)
    Returns   : np.ndarray of shape (H, W, 3) in BGR channel order
    """
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_image: np.ndarray):
    """Convert an OpenCV BGR numpy array back to a PIL image.

    bgr_image : np.ndarray of shape (H, W, 3) in BGR channel order
    Returns   : a PIL.Image.Image in RGB mode
    """
    from PIL import Image  # lazy — only needed when handing images back to Chandra

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_config_overrides(overrides: Optional[dict]) -> GridDetectionConfig:
    """Build a GridDetectionConfig from grid_detection.CONFIG plus optional overrides.

    overrides : dict of {field_name: value} applied onto a copy of the defaults;
                None or empty returns the default CONFIG unchanged.
    Returns   : a GridDetectionConfig instance.
    """
    if not overrides:
        return CONFIG
    base = asdict(CONFIG)
    base.update(overrides)
    return GridDetectionConfig(**base)


# ---------------------------------------------------------------------------
# In-memory grid pipeline (detection + removal + all masks)
# ---------------------------------------------------------------------------
def run_grid_pipeline(
    bgr_image: np.ndarray,
    config: Optional[GridDetectionConfig] = None,
) -> dict:
    """Run the full grid_detection.py grid pipeline on one page image, in memory.

    Reuses grid_detection.py.detect_dense_grid_regions + remove_dense_grid_lines. Bangla
    matras and normal tables are protected by the same rules used in grid_detection.py.

    bgr_image : source page image as an OpenCV BGR numpy array
    config    : GridDetectionConfig (defaults to grid_detection.CONFIG when None)
    Returns   : a bundle dict with everything needed to re-OCR or to save the
                grid_detection.py artifacts:
                  "source_bgr", "cleaned", "removed_mask", "line_mask",
                  "horizontal_mask", "vertical_mask", "dense_regions",
                  "rejected_regions", "config", and "report" (JSON-serializable:
                  page size, region counts, regions_found, grid_lines_removed,
                  the regions, and the config).
                "cleaned" is a copy of the input when no dense grid is found, so
                callers can treat the return value uniformly.
    Saves     : nothing (in-memory only; use save_test_artifacts to persist).
    """
    cfg = config or CONFIG

    (
        dense_regions,
        rejected_regions,
        line_mask,
        horizontal_mask,
        vertical_mask,
    ) = detect_dense_grid_regions(image=bgr_image, config=cfg)

    if dense_regions:
        cleaned, removed_mask = remove_dense_grid_lines(
            image=bgr_image,
            dense_regions=dense_regions,
            horizontal_mask=horizontal_mask,
            vertical_mask=vertical_mask,
            config=cfg,
        )
        grid_lines_removed = True
    else:
        # No dense grid detected — return an untouched copy and an empty removal
        # mask so callers/artifacts stay uniform.
        cleaned = bgr_image.copy()
        removed_mask = np.zeros(line_mask.shape, dtype=np.uint8)
        grid_lines_removed = False

    report = {
        "page_width": int(bgr_image.shape[1]),
        "page_height": int(bgr_image.shape[0]),
        "dense_grid_region_count": len(dense_regions),
        "rejected_region_count": len(rejected_regions),
        "regions_found": bool(dense_regions),
        "grid_lines_removed": grid_lines_removed,
        "dense_grid_regions": dense_regions,
        "rejected_regions": rejected_regions,
        "config": asdict(cfg),
    }

    return {
        "source_bgr": bgr_image,
        "cleaned": cleaned,
        "removed_mask": removed_mask,
        "line_mask": line_mask,
        "horizontal_mask": horizontal_mask,
        "vertical_mask": vertical_mask,
        "dense_regions": dense_regions,
        "rejected_regions": rejected_regions,
        "config": cfg,
        "report": report,
    }


def clean_page_image(
    bgr_image: np.ndarray,
    config: Optional[GridDetectionConfig] = None,
) -> tuple[np.ndarray, dict]:
    """Convenience wrapper: return just the cleaned image and its JSON report.

    Delegates to run_grid_pipeline(). Kept for the simple "clean + re-OCR" path.

    bgr_image : source page image as an OpenCV BGR numpy array
    config    : GridDetectionConfig (defaults to grid_detection.CONFIG when None)
    Returns   : (cleaned_bgr, report) — see run_grid_pipeline for report shape.
    Saves     : nothing.
    """
    pipeline = run_grid_pipeline(bgr_image, config)
    return pipeline["cleaned"], pipeline["report"]


# ---------------------------------------------------------------------------
# Artifact saving (the same 8 files grid_detection.py.process_image writes)
# ---------------------------------------------------------------------------
def save_test_artifacts(pipeline: dict, out_dir: Path, image_ref: str) -> None:
    """Write the full set of debug artifacts grid_detection.py produces for one page.

    Mirrors grid_detection.py.process_image() exactly (same filenames), reusing grid_detection.py's
    draw_regions / draw_dense_and_rejected_regions / save_json.

    pipeline  : the bundle returned by run_grid_pipeline()
    out_dir   : directory to write the 8 files into (created if missing)
    image_ref : label recorded as "image_path" inside dense_grid_regions.json
    Returns   : None
    Saves     : out_dir/{dense_grid_regions.json, dense_grid_regions_marked.png,
                dense_and_rejected_regions_debug.png, cleaned_dense_grids_only.png,
                removed_dense_grid_line_mask.png, raw_line_mask.png,
                horizontal_line_mask.png, vertical_line_mask.png}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = pipeline["source_bgr"]
    dense_regions = pipeline["dense_regions"]
    rejected_regions = pipeline["rejected_regions"]

    # Red boxes around dense grids that were cleaned.
    draw_regions(
        image=source,
        regions=dense_regions,
        output_path=out_dir / "dense_grid_regions_marked.png",
        color=(0, 0, 255),
    )

    # Red = cleaned, Blue = rejected table/line candidates.
    draw_dense_and_rejected_regions(
        image=source,
        dense_regions=dense_regions,
        rejected_regions=rejected_regions,
        output_path=out_dir / "dense_and_rejected_regions_debug.png",
    )

    cv2.imwrite(str(out_dir / "cleaned_dense_grids_only.png"), pipeline["cleaned"])
    cv2.imwrite(str(out_dir / "removed_dense_grid_line_mask.png"), pipeline["removed_mask"])
    cv2.imwrite(str(out_dir / "raw_line_mask.png"), pipeline["line_mask"])
    cv2.imwrite(str(out_dir / "horizontal_line_mask.png"), pipeline["horizontal_mask"])
    cv2.imwrite(str(out_dir / "vertical_line_mask.png"), pipeline["vertical_mask"])

    test_save_json(
        image_path=image_ref,
        image=source,
        dense_regions=dense_regions,
        rejected_regions=rejected_regions,
        config=pipeline["config"],
        output_path=out_dir / "dense_grid_regions.json",
    )


if __name__ == "__main__":
    # Tiny offline smoke test: a blank white page has no dense grid, so cleaning
    # must be a no-op that still returns a valid report and full pipeline bundle.
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    out, report = clean_page_image(blank)
    assert out.shape == blank.shape
    assert report["regions_found"] is False
    assert report["grid_lines_removed"] is False

    pipeline = run_grid_pipeline(blank)
    for key in ("cleaned", "removed_mask", "line_mask", "horizontal_mask",
                "vertical_mask", "dense_regions", "rejected_regions", "report"):
        assert key in pipeline, f"missing pipeline key {key!r}"
    assert pipeline["removed_mask"].shape == blank.shape[:2]
    print("[ok] grid_clean no-op on blank page")
    print("[ok] run_grid_pipeline bundle shape")
    print("ALL GRID-CLEAN SMOKE CHECKS PASSED")

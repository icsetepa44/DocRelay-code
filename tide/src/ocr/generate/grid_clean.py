from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import sys as _sys
_DU = Path(__file__).resolve().parents[3]
if str(_DU) not in _sys.path:
    _sys.path.insert(0, str(_DU))

from src.ocr.generate.grid_detection import (
    CONFIG,
    GridDetectionConfig,
    detect_dense_grid_regions,
    draw_dense_and_rejected_regions,
    draw_regions,
    remove_dense_grid_lines,
    save_json as test_save_json,
)


def pil_to_bgr(pil_image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_image: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_config_overrides(overrides: Optional[dict]) -> GridDetectionConfig:
    if not overrides:
        return CONFIG
    base = asdict(CONFIG)
    base.update(overrides)
    return GridDetectionConfig(**base)


def run_grid_pipeline(
    bgr_image: np.ndarray,
    config: Optional[GridDetectionConfig] = None,
) -> dict:
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
    pipeline = run_grid_pipeline(bgr_image, config)
    return pipeline["cleaned"], pipeline["report"]


def save_test_artifacts(pipeline: dict, out_dir: Path, image_ref: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = pipeline["source_bgr"]
    dense_regions = pipeline["dense_regions"]
    rejected_regions = pipeline["rejected_regions"]

    draw_regions(
        image=source,
        regions=dense_regions,
        output_path=out_dir / "dense_grid_regions_marked.png",
        color=(0, 0, 255),
    )

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

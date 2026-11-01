import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


# ==================================================
# USER VARIABLES
# ==================================================
# IMAGE_PATH = r"data\work\images\20160408_Finance_Act_2013\page_0036.png"
IMAGE_PATH = r"data\work\images\20240606_209-AIN_2024_61_Customs\page_0018.png"
OUTPUT_DIR = "outputs/grid_cleanup_2"


# ==================================================
# CONFIG
# ==================================================

@dataclass
class GridDetectionConfig:
    # Adaptive threshold
    adaptive_block_size: int = 31
    adaptive_c: int = 15

    # Line extraction kernel scaling
    horizontal_kernel_divisor: int = 70
    vertical_kernel_divisor: int = 65

    # Minimum kernel sizes
    min_horizontal_kernel: int = 20
    min_vertical_kernel: int = 18

    # Small dilation only to connect broken grid intersections
    connect_dilation_kernel: tuple = (3, 3)
    connect_dilation_iterations: int = 1

    # Basic candidate region filters
    min_region_width: int = 120
    min_region_height: int = 18
    min_region_area: int = 800

    # Dense-grid rules
    # These are the main protections against normal tables.
    min_vertical_lines: int = 8
    min_horizontal_lines: int = 2
    min_estimated_cells: int = 12
    min_intersections: int = 12
    min_narrow_cell_gaps: int = 8

    # A dense character grid has many small repeated cell gaps.
    # These are dynamic using page width, so it works better across resolutions.
    min_cell_width_page_ratio: float = 0.003
    max_cell_width_page_ratio: float = 0.050

    # Absolute backup limits
    min_cell_width_px: int = 5
    max_cell_width_px: int = 90

    # Projection thresholds for detecting line positions inside a region
    horizontal_line_coverage_ratio: float = 0.25
    vertical_line_coverage_ratio: float = 0.35

    min_horizontal_line_span_px: int = 30
    min_vertical_line_span_px: int = 8

    # Padding around bbox
    bbox_padding: int = 2

    # Slightly expand line mask before removal
    removal_dilation_kernel: tuple = (2, 2)
    removal_dilation_iterations: int = 1
    
        # Safe removal rules
    # A horizontal component is removed only if it touches enough vertical grid lines.
    min_confirmed_horizontal_intersections: int = 3

    # A vertical component is removed only if it touches enough horizontal grid lines.
    min_confirmed_vertical_intersections: int = 2

    # Ignore tiny text-like components
    min_horizontal_component_width: int = 20
    min_vertical_component_height: int = 12

    # Expansion used only for intersection checking
    intersection_check_dilation_kernel: tuple = (3, 3)


CONFIG = GridDetectionConfig()


# ==================================================
# IMAGE LOADING
# ==================================================

def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return image


def get_image_output_dir(image_path: str, base_output_dir: str) -> Path:
    image_name = Path(image_path).stem
    return Path(base_output_dir) / image_name


# ==================================================
# THRESHOLDING
# ==================================================

def make_binary_inverse(image: np.ndarray, config: GridDetectionConfig) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    return binary_inv


# ==================================================
# LINE MASK EXTRACTION
# ==================================================

def extract_grid_line_masks(
    image: np.ndarray,
    config: GridDetectionConfig,
):
    """
    Extract horizontal and vertical ruled lines from the page.

    This only creates masks.
    It does not remove anything yet.
    """

    binary_inv = make_binary_inverse(image, config)

    page_h, page_w = binary_inv.shape

    horizontal_kernel_len = max(
        config.min_horizontal_kernel,
        page_w // config.horizontal_kernel_divisor,
    )

    vertical_kernel_len = max(
        config.min_vertical_kernel,
        page_h // config.vertical_kernel_divisor,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_len, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_len),
    )

    horizontal_mask = cv2.morphologyEx(
        binary_inv,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    vertical_mask = cv2.morphologyEx(
        binary_inv,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    line_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config.connect_dilation_kernel,
    )

    connected_line_mask = cv2.dilate(
        line_mask,
        connect_kernel,
        iterations=config.connect_dilation_iterations,
    )

    return connected_line_mask, horizontal_mask, vertical_mask


# ==================================================
# BBOX HELPERS
# ==================================================

def clamp_bbox(x1, y1, x2, y2, page_w, page_h, padding=0):
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(page_w, int(x2) + padding)
    y2 = min(page_h, int(y2) + padding)

    return x1, y1, x2, y2


# ==================================================
# 1D LINE CLUSTERING
# ==================================================

def find_projection_clusters(projection, threshold):
    """
    Finds clusters in a 1D projection.

    Example:
    For vertical lines, we project the vertical mask along the y-axis.
    Columns with many white pixels indicate vertical line positions.

    Returns:
    - centers: center position of each line cluster
    - spans: start/end positions of each cluster
    """

    active = projection >= threshold

    centers = []
    spans = []

    start = None

    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx

        elif not is_active and start is not None:
            end = idx - 1
            center = (start + end) // 2
            centers.append(center)
            spans.append((start, end))
            start = None

    if start is not None:
        end = len(active) - 1
        center = (start + end) // 2
        centers.append(center)
        spans.append((start, end))

    return centers, spans


# ==================================================
# DENSE GRID ANALYSIS
# ==================================================

def count_intersections(horizontal_crop, vertical_crop):
    """
    Counts intersections between horizontal and vertical ruled lines.

    A single horizontal line will have zero or very few intersections.
    A real rectangular grid has many intersections.
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    h_dilated = cv2.dilate(horizontal_crop, kernel, iterations=1)
    v_dilated = cv2.dilate(vertical_crop, kernel, iterations=1)

    intersection_mask = cv2.bitwise_and(h_dilated, v_dilated)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        intersection_mask,
        connectivity=8,
    )

    count = 0

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]

        if area >= 1:
            count += 1

    return count, intersection_mask


def analyze_dense_grid_candidate(
    bbox,
    horizontal_mask,
    vertical_mask,
    page_w,
    page_h,
    config: GridDetectionConfig,
):
    """
    Decides whether a bbox is a dense form-grid region.

    This is the main anti-normal-table filter.
    """

    x1, y1, x2, y2 = bbox

    region_w = x2 - x1
    region_h = y2 - y1

    horizontal_crop = horizontal_mask[y1:y2, x1:x2]
    vertical_crop = vertical_mask[y1:y2, x1:x2]

    # ----------------------------------------------
    # Count horizontal line positions
    # ----------------------------------------------

    horizontal_projection = np.count_nonzero(horizontal_crop, axis=1)

    horizontal_threshold = max(
        config.min_horizontal_line_span_px,
        int(region_w * config.horizontal_line_coverage_ratio),
    )

    horizontal_positions, horizontal_spans = find_projection_clusters(
        horizontal_projection,
        horizontal_threshold,
    )

    # ----------------------------------------------
    # Count vertical line positions
    # ----------------------------------------------

    vertical_projection = np.count_nonzero(vertical_crop, axis=0)

    vertical_threshold = max(
        config.min_vertical_line_span_px,
        int(region_h * config.vertical_line_coverage_ratio),
    )

    vertical_positions, vertical_spans = find_projection_clusters(
        vertical_projection,
        vertical_threshold,
    )

    vertical_line_count = len(vertical_positions)
    horizontal_line_count = len(horizontal_positions)

    # ----------------------------------------------
    # Count estimated rectangular cells
    # ----------------------------------------------

    estimated_cells = max(0, vertical_line_count - 1) * max(
        0,
        horizontal_line_count - 1,
    )

    # ----------------------------------------------
    # Count narrow repeated cell gaps
    # ----------------------------------------------

    vertical_gaps = []

    for i in range(1, len(vertical_positions)):
        gap = vertical_positions[i] - vertical_positions[i - 1]
        vertical_gaps.append(gap)

    min_cell_width = max(
        config.min_cell_width_px,
        int(page_w * config.min_cell_width_page_ratio),
    )

    max_cell_width = max(
        config.max_cell_width_px,
        int(page_w * config.max_cell_width_page_ratio),
    )

    narrow_cell_gaps = [
        gap for gap in vertical_gaps
        if min_cell_width <= gap <= max_cell_width
    ]

    if narrow_cell_gaps:
        median_cell_width = float(np.median(narrow_cell_gaps))
    else:
        median_cell_width = None

    # ----------------------------------------------
    # Count line intersections
    # ----------------------------------------------

    intersection_count, intersection_mask = count_intersections(
        horizontal_crop,
        vertical_crop,
    )

    # ----------------------------------------------
    # Final dense-grid decision
    # ----------------------------------------------

    reasons = []

    if vertical_line_count < config.min_vertical_lines:
        reasons.append(
            f"vertical_line_count {vertical_line_count} < {config.min_vertical_lines}"
        )

    if horizontal_line_count < config.min_horizontal_lines:
        reasons.append(
            f"horizontal_line_count {horizontal_line_count} < {config.min_horizontal_lines}"
        )

    if estimated_cells < config.min_estimated_cells:
        reasons.append(
            f"estimated_cells {estimated_cells} < {config.min_estimated_cells}"
        )

    if intersection_count < config.min_intersections:
        reasons.append(
            f"intersection_count {intersection_count} < {config.min_intersections}"
        )

    if len(narrow_cell_gaps) < config.min_narrow_cell_gaps:
        reasons.append(
            f"narrow_cell_gaps {len(narrow_cell_gaps)} < {config.min_narrow_cell_gaps}"
        )

    is_dense_grid = len(reasons) == 0

    debug_info = {
        "is_dense_grid": is_dense_grid,
        "rejection_reasons": reasons,
        "vertical_line_count": vertical_line_count,
        "horizontal_line_count": horizontal_line_count,
        "estimated_cells": int(estimated_cells),
        "intersection_count": int(intersection_count),
        "narrow_cell_gap_count": int(len(narrow_cell_gaps)),
        "median_cell_width": median_cell_width,
        "min_cell_width": int(min_cell_width),
        "max_cell_width": int(max_cell_width),
        "vertical_positions": [int(v) + x1 for v in vertical_positions],
        "horizontal_positions": [int(h) + y1 for h in horizontal_positions],
    }

    return is_dense_grid, debug_info, intersection_mask


# ==================================================
# DENSE GRID REGION DETECTION
# ==================================================

def detect_dense_grid_regions(
    image: np.ndarray,
    config: GridDetectionConfig,
):
    """
    Detects only dense form-grid regions.

    It rejects:
    - single horizontal lines
    - underlines
    - normal simple tables
    - boxes with only a few rows/columns
    """

    line_mask, horizontal_mask, vertical_mask = extract_grid_line_masks(
        image,
        config,
    )

    page_h, page_w = line_mask.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask,
        connectivity=8,
    )

    dense_regions = []
    rejected_regions = []

    for label_idx in range(1, num_labels):
        x, y, bw, bh, component_area = stats[label_idx]

        # ------------------------------------------
        # Basic shape filter
        # ------------------------------------------

        if bw < config.min_region_width:
            continue

        if bh < config.min_region_height:
            continue

        if component_area < config.min_region_area:
            continue

        x1, y1, x2, y2 = clamp_bbox(
            x,
            y,
            x + bw,
            y + bh,
            page_w,
            page_h,
            padding=config.bbox_padding,
        )

        bbox = [x1, y1, x2, y2]

        # ------------------------------------------
        # Dense-grid filter
        # ------------------------------------------

        is_dense_grid, debug_info, _ = analyze_dense_grid_candidate(
            bbox=bbox,
            horizontal_mask=horizontal_mask,
            vertical_mask=vertical_mask,
            page_w=page_w,
            page_h=page_h,
            config=config,
        )

        region_w = x2 - x1
        region_h = y2 - y1

        region_record = {
            "id": "",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "width": int(region_w),
            "height": int(region_h),
            "area": int(region_w * region_h),
            "component_area": int(component_area),
            "aspect_ratio": round(region_w / max(region_h, 1), 2),
            "dense_grid_debug": debug_info,
        }

        if is_dense_grid:
            dense_regions.append(region_record)
        else:
            rejected_regions.append(region_record)

    dense_regions = sorted(dense_regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    rejected_regions = sorted(rejected_regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    for idx, region in enumerate(dense_regions, start=1):
        region["id"] = f"dense_grid_region_{idx}"

    for idx, region in enumerate(rejected_regions, start=1):
        region["id"] = f"rejected_region_{idx}"

    return dense_regions, rejected_regions, line_mask, horizontal_mask, vertical_mask


def build_confirmed_grid_removal_mask(
    horizontal_crop: np.ndarray,
    vertical_crop: np.ndarray,
    config: GridDetectionConfig,
):
    """
    Builds a safe removal mask.

    It removes:
    - horizontal grid lines that intersect several vertical grid lines
    - vertical grid lines that intersect horizontal grid lines

    It does NOT remove:
    - Bangla matras inside the grid
    - standalone horizontal strokes
    - text-like horizontal/vertical components
    """

    crop_h, crop_w = horizontal_crop.shape

    confirmed_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    intersection_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config.intersection_check_dilation_kernel,
    )

    vertical_for_check = cv2.dilate(
        vertical_crop,
        intersection_kernel,
        iterations=1,
    )

    horizontal_for_check = cv2.dilate(
        horizontal_crop,
        intersection_kernel,
        iterations=1,
    )

    # ==================================================
    # Confirm horizontal grid lines
    # ==================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        horizontal_crop,
        connectivity=8,
    )

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]

        if w < config.min_horizontal_component_width:
            continue

        component_mask = np.zeros_like(horizontal_crop)
        component_mask[labels == label_idx] = 255

        # Check where this horizontal component intersects vertical lines.
        intersection = cv2.bitwise_and(
            cv2.dilate(component_mask, intersection_kernel, iterations=1),
            vertical_for_check,
        )

        # Count distinct vertical intersection columns.
        x_projection = np.count_nonzero(intersection, axis=0)

        intersection_columns, _ = find_projection_clusters(
            x_projection,
            threshold=1,
        )

        intersection_count = len(intersection_columns)

        if intersection_count >= config.min_confirmed_horizontal_intersections:
            confirmed_mask[labels == label_idx] = 255

    # ==================================================
    # Confirm vertical grid lines
    # ==================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vertical_crop,
        connectivity=8,
    )

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]

        if h < config.min_vertical_component_height:
            continue

        component_mask = np.zeros_like(vertical_crop)
        component_mask[labels == label_idx] = 255

        # Check where this vertical component intersects horizontal lines.
        intersection = cv2.bitwise_and(
            cv2.dilate(component_mask, intersection_kernel, iterations=1),
            horizontal_for_check,
        )

        # Count distinct horizontal intersection rows.
        y_projection = np.count_nonzero(intersection, axis=1)

        intersection_rows, _ = find_projection_clusters(
            y_projection,
            threshold=1,
        )

        intersection_count = len(intersection_rows)

        if intersection_count >= config.min_confirmed_vertical_intersections:
            confirmed_mask[labels == label_idx] = 255

    # Slightly expand the final confirmed grid-line mask.
    removal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config.removal_dilation_kernel,
    )

    confirmed_mask = cv2.dilate(
        confirmed_mask,
        removal_kernel,
        iterations=config.removal_dilation_iterations,
    )

    return confirmed_mask


# ==================================================
# DRAW REGIONS
# ==================================================

def draw_regions(
    image: np.ndarray,
    regions: list,
    output_path: Path,
    color=(0, 0, 255),
):
    marked = image.copy()

    for idx, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = region["bbox"]

        cv2.rectangle(
            marked,
            (x1, y1),
            (x2, y2),
            color,
            thickness=3,
        )

        cv2.putText(
            marked,
            str(idx),
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), marked)


def draw_dense_and_rejected_regions(
    image: np.ndarray,
    dense_regions: list,
    rejected_regions: list,
    output_path: Path,
):
    """
    Debug image:
    - Red boxes = dense grids that will be cleaned
    - Blue boxes = rejected table/line candidates that will NOT be cleaned
    """

    marked = image.copy()

    # Rejected regions in blue
    for idx, region in enumerate(rejected_regions, start=1):
        x1, y1, x2, y2 = region["bbox"]

        cv2.rectangle(
            marked,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            thickness=2,
        )

    # Dense regions in red
    for idx, region in enumerate(dense_regions, start=1):
        x1, y1, x2, y2 = region["bbox"]

        cv2.rectangle(
            marked,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            thickness=3,
        )

        cv2.putText(
            marked,
            str(idx),
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), marked)


# ==================================================
# REMOVE GRID LINES
# ==================================================

def remove_dense_grid_lines(
    image: np.ndarray,
    dense_regions: list,
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    config: GridDetectionConfig,
):
    """
    Removes only confirmed rectangular grid-line pixels inside dense grid regions.

    This protects Bangla matras because a matra may look like a horizontal line,
    but it does not intersect repeated vertical grid lines.
    """

    cleaned = image.copy()

    page_h, page_w = horizontal_mask.shape
    removed_mask = np.zeros((page_h, page_w), dtype=np.uint8)

    for region in dense_regions:
        x1, y1, x2, y2 = region["bbox"]

        x1, y1, x2, y2 = clamp_bbox(
            x1,
            y1,
            x2,
            y2,
            page_w,
            page_h,
            padding=1,
        )

        horizontal_crop = horizontal_mask[y1:y2, x1:x2]
        vertical_crop = vertical_mask[y1:y2, x1:x2]

        confirmed_grid_mask = build_confirmed_grid_removal_mask(
            horizontal_crop=horizontal_crop,
            vertical_crop=vertical_crop,
            config=config,
        )

        crop = cleaned[y1:y2, x1:x2]

        # Critical fix:
        # whiten only confirmed rectangular grid-line pixels.
        # Do NOT whiten raw line_mask.
        crop[confirmed_grid_mask > 0] = 255

        cleaned[y1:y2, x1:x2] = crop

        removed_mask[y1:y2, x1:x2] = cv2.bitwise_or(
            removed_mask[y1:y2, x1:x2],
            confirmed_grid_mask,
        )

    return cleaned, removed_mask


# ==================================================
# SAVE JSON
# ==================================================

def save_json(
    image_path: str,
    image: np.ndarray,
    dense_regions: list,
    rejected_regions: list,
    config: GridDetectionConfig,
    output_path: Path,
):
    result = {
        "image_path": image_path,
        "page_width": int(image.shape[1]),
        "page_height": int(image.shape[0]),
        "dense_grid_region_count": len(dense_regions),
        "rejected_region_count": len(rejected_regions),
        "dense_grid_regions": dense_regions,
        "rejected_regions": rejected_regions,
        "config": asdict(config),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ==================================================
# MAIN PROCESS
# ==================================================

def process_image():
    output_dir = get_image_output_dir(
        image_path=IMAGE_PATH,
        base_output_dir=OUTPUT_DIR,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(IMAGE_PATH)

    (
        dense_regions,
        rejected_regions,
        line_mask,
        horizontal_mask,
        vertical_mask,
    ) = detect_dense_grid_regions(
        image=image,
        config=CONFIG,
    )

    json_path = output_dir / "dense_grid_regions.json"
    dense_marked_path = output_dir / "dense_grid_regions_marked.png"
    dense_and_rejected_path = output_dir / "dense_and_rejected_regions_debug.png"
    cleaned_path = output_dir / "cleaned_dense_grids_only.png"
    removed_mask_path = output_dir / "removed_dense_grid_line_mask.png"
    raw_line_mask_path = output_dir / "raw_line_mask.png"
    horizontal_mask_path = output_dir / "horizontal_line_mask.png"
    vertical_mask_path = output_dir / "vertical_line_mask.png"

    # Red boxes only around dense grids that will be cleaned
    draw_regions(
        image=image,
        regions=dense_regions,
        output_path=dense_marked_path,
        color=(0, 0, 255),
    )

    # Red = cleaned, Blue = rejected
    draw_dense_and_rejected_regions(
        image=image,
        dense_regions=dense_regions,
        rejected_regions=rejected_regions,
        output_path=dense_and_rejected_path,
    )

    cleaned, removed_mask = remove_dense_grid_lines(
        image=image,
        dense_regions=dense_regions,
        horizontal_mask=horizontal_mask,
        vertical_mask=vertical_mask,
        config=CONFIG,
    )

    cv2.imwrite(str(cleaned_path), cleaned)
    cv2.imwrite(str(removed_mask_path), removed_mask)
    cv2.imwrite(str(raw_line_mask_path), line_mask)
    cv2.imwrite(str(horizontal_mask_path), horizontal_mask)
    cv2.imwrite(str(vertical_mask_path), vertical_mask)

    save_json(
        image_path=IMAGE_PATH,
        image=image,
        dense_regions=dense_regions,
        rejected_regions=rejected_regions,
        config=CONFIG,
        output_path=json_path,
    )

    print(f"Image: {Path(IMAGE_PATH).name}")
    print(f"Output directory: {output_dir}")
    print(f"Dense grid regions cleaned: {len(dense_regions)}")
    print(f"Rejected table/line candidates: {len(rejected_regions)}")
    print("Saved files:")
    print(json_path)
    print(dense_marked_path)
    print(dense_and_rejected_path)
    print(cleaned_path)
    print(removed_mask_path)
    print(raw_line_mask_path)
    print(horizontal_mask_path)
    print(vertical_mask_path)


if __name__ == "__main__":
    process_image()
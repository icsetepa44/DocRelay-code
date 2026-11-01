from __future__ import annotations

from typing import Dict, List


def _norm_len(s: str) -> int:
    return len("".join(str(s).split()))


def coverage_ratio(block_texts: List[str], ocr_text: str) -> float:
    ocr = _norm_len(ocr_text)
    if ocr == 0:
        return 1.0
    return sum(_norm_len(t) for t in block_texts) / ocr


def analyze_underproduction(n_regions: int, block_texts: List[str], ocr_text: str, *,
                            min_regions: int = 5, block_ratio: float = 0.25,
                            min_coverage: float = 0.3) -> Dict:
    n_blocks = len(block_texts)
    block_cap = max(2, round(block_ratio * n_regions))
    cov = coverage_ratio(block_texts, ocr_text)
    under = (n_regions >= min_regions and n_blocks <= block_cap and cov < min_coverage)
    return {
        "n_regions": n_regions,
        "n_blocks": n_blocks,
        "block_cap": block_cap,
        "ocr_chars": _norm_len(ocr_text),
        "block_chars": sum(_norm_len(t) for t in block_texts),
        "coverage": round(cov, 4),
        "under_produced": bool(under),
    }

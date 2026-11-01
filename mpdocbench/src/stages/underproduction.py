"""Page under-production detector (Track 2 page-drop guard).

Some extractors (observed on gemma) bail out mid-page: after resolving the incoming continuation they
emit only a block or two of near-empty content and stop, ignoring the rest of a text-rich page. The
JSON is well-formed, so the orchestrator's existing exception/zero-block guards never fire — the page's
content is silently lost (and, downstream, its whole rank can desync the reading-order metric).

This module is a pure diagnostic: given the region count (the extractor's own upstream segmentation
prior), the emitted block texts, and the page's OCR text, it decides whether the page was
under-produced. A page trips the guard only when ALL of:
  - the region analyzer found at least ``min_regions`` regions (the page really has many elements),
  - the extractor emitted no more than ``max(2, round(block_ratio * n_regions))`` blocks, AND
  - those blocks cover less than ``min_coverage`` of the page's OCR text.
The coverage clause protects legitimately coarse pages (one big table, or over-split regions collapsed
into a couple of blocks that still capture the text); the region clause protects near-blank / furniture
pages. The orchestrator writes the record every page (``page_underproduction.json``) and, when the retry
toggle is on, re-extracts flagged pages with a corrective nudge. Never raises — a diagnostic must never
break a run.
"""

from __future__ import annotations

from typing import Dict, List


def _norm_len(s: str) -> int:
    """Whitespace-insensitive character count (so OCR/extraction spacing differences don't skew coverage)."""
    return len("".join(str(s).split()))


def coverage_ratio(block_texts: List[str], ocr_text: str) -> float:
    """Fraction of the page's OCR characters captured by the emitted block texts, in [0, 1+].

    ``block_texts`` should already be table-aware (the caller flattens table cells via ``block_text``),
    so a page that is legitimately one big table still scores high coverage. Empty OCR -> 1.0 (nothing
    to capture, never under-produced)."""
    ocr = _norm_len(ocr_text)
    if ocr == 0:
        return 1.0
    return sum(_norm_len(t) for t in block_texts) / ocr


def analyze_underproduction(n_regions: int, block_texts: List[str], ocr_text: str, *,
                            min_regions: int = 5, block_ratio: float = 0.25,
                            min_coverage: float = 0.3) -> Dict:
    """Pure detector. Returns a JSON-serializable record; ``under_produced`` is the headline flag.

    See the module docstring for the rule. Thresholds are passed in so the orchestrator can source them
    from ``pipeline.yaml``."""
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

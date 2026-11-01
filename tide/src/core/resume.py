"""
Page-level resume for interrupted runs.

An interrupted (doc, model) pair has some pages already persisted under
``run_dir/<doc>/<label>/pages/<page4>/``. This module reconstructs the state the per-page
loop would have had at the first not-yet-complete page:

  * the rehydrated ``PageExtraction`` prefix (so the assembler still sees the whole document),
  * the seed ``CarryState`` (threaded page -> page; never reloaded by the loop itself),
  * the ``StitchIdAllocator`` counter (restored so new stitch ids can't collide with issued ones),
  * best-effort ``page_summaries`` (so run_summary.json / final_validator stay ~accurate).

``orchestrator.run_document`` uses this to skip re-extracting finished pages and continue.

Pure & deterministic: only ``json``/``pathlib`` + contracts. No LLM calls, no image loads,
and — via ``run_store.page_dir_path`` — no directory creation while scanning.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData, PageExtraction
from src.storage import run_store

# The two artifacts resume correctness depends on. In _persist_page, carry_state.json is written
# AFTER extraction.json, so a parseable carry_state.json guarantees extraction.json was fully
# flushed; the trailing diagnostic ledgers don't affect resume and are dropped by the tail clean.
_REQUIRED_PAGE_FILES = ("extraction.json", "carry_state.json")


def _load_json(path: Path) -> Optional[Any]:
    """Parse a JSON file, returning None if it's missing or torn (half-written)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def page_complete(page_dir: Path) -> bool:
    """A persisted page is resumable iff extraction.json AND carry_state.json both exist and
    parse as JSON (a torn/half-written file fails json.loads -> incomplete)."""
    return all(_load_json(page_dir / name) is not None for name in _REQUIRED_PAGE_FILES)


def _stitch_seq(stitch_id_value: Optional[str]) -> int:
    """Parse the numeric seq out of a stitch id ``{document_id}:s{seq:06d}``. rsplit tolerates a
    document_id that itself contains ':'. Returns 0 for None / malformed."""
    if not stitch_id_value:
        return 0
    tail = stitch_id_value.rsplit(":", 1)[-1]      # e.g. 's000012'
    if not tail.startswith("s") or not tail[1:].isdigit():
        return 0
    return int(tail[1:])


def _max_stitch_seq(extractions: List[PageExtraction], carry: CarryState) -> int:
    """Highest stitch seq already issued: scan every rehydrated block (captures ids of units that
    have since CLOSED) and every open carry unit (captures currently-OPEN ids). allocator.next()
    increments-then-returns, so seeding start=this makes the next id max+1 — no collision."""
    best = 0
    for pe in extractions:
        for b in pe.blocks:
            best = max(best, _stitch_seq(b.stitch_id))
    for u in carry.open_units:
        best = max(best, _stitch_seq(u.stitch_id))
    return best


def _reconstruct_summary(page_dir: Path, pe: PageExtraction, carry: CarryState,
                         page: PageData, min_chars_nontrivial: int) -> Dict[str, Any]:
    """Best-effort rebuild of one per-page summary dict (mirrors orchestrator's page_summaries
    entry). Load-bearing fields (empty_nontrivial, open_units_after) are recomputed exactly;
    diagnostic counts come from the per-page ledgers when present, else default to 0."""
    nontrivial_empty = (not pe.blocks
                        and len(page.markdown.strip()) >= min_chars_nontrivial)
    validation = _load_json(page_dir / "validation.json") or {}
    judge = _load_json(page_dir / "continuation_judge.json") or {}
    rejection = _load_json(page_dir / "carry_rejection.json") or {}
    over_seg = _load_json(page_dir / "over_segmentation.json") or {}
    return {
        "page_number": page.page_number,
        "num_blocks": len(pe.blocks),
        "validation_status": ("empty_page" if nontrivial_empty
                              else (validation.get("status") or "reloaded")),
        "empty_nontrivial": nontrivial_empty,
        "open_units_after": [u.stitch_id for u in carry.open_units],
        "judge_invoked": bool(judge.get("judge_invoked")),
        "judged_count": len(judge.get("judged_block_ids") or []),
        "carry_rejections": rejection.get("rejected_count", 0),
        "over_seg_runs": over_seg.get("run_count", 0),
    }


@dataclass
class ResumeState:
    """State to seed run_document at the first not-yet-complete page."""

    index: int                                    # loop resume index == len(complete prefix)
    extractions: List[PageExtraction] = field(default_factory=list)
    carry: CarryState = field(default_factory=CarryState.empty)
    allocator_start: int = 0
    page_summaries: List[Dict[str, Any]] = field(default_factory=list)
    resume_page_number: Optional[int] = None      # pages[index].page_number, or None if all done


def compute_resume_state(run_dir: Path, doc_name: str, model_label: Optional[str],
                         pages: List[PageData], *, min_chars_nontrivial: int) -> ResumeState:
    """Scan the (already page_start/page_end-sliced) ``pages`` list and rehydrate the longest
    CONTIGUOUS prefix of complete pages, keyed by ``page.page_number``. Read-only — never creates
    dirs. ``index == 0`` means nothing resumable (a fresh run)."""
    extractions: List[PageExtraction] = []
    summaries: List[Dict[str, Any]] = []
    carry = CarryState.empty()
    index = 0
    for page in pages:
        pdir = run_store.page_dir_path(run_dir, doc_name, page.page_number, model_label)
        if not page_complete(pdir):
            break
        pe = PageExtraction.from_dict(_load_json(pdir / "extraction.json"))
        carry = CarryState.from_dict(_load_json(pdir / "carry_state.json"))
        extractions.append(pe)
        summaries.append(_reconstruct_summary(pdir, pe, carry, page, min_chars_nontrivial))
        index += 1
    return ResumeState(
        index=index,
        extractions=extractions,
        carry=carry,
        allocator_start=_max_stitch_seq(extractions, carry),
        page_summaries=summaries,
        resume_page_number=(pages[index].page_number if index < len(pages) else None),
    )


def clean_incomplete_tail(run_dir: Path, doc_name: str, model_label: Optional[str],
                          pages: List[PageData], index: int) -> List[int]:
    """Remove page dirs for ``pages[index:]`` (the torn resume page + anything after it) so a
    reprocessed page never inherits stale/torn files (e.g. an orphan region_analysis.json from a
    prior region-mode run). Returns the page_numbers removed."""
    removed: List[int] = []
    for page in pages[index:]:
        pdir = run_store.page_dir_path(run_dir, doc_name, page.page_number, model_label)
        if pdir.exists():
            shutil.rmtree(pdir, ignore_errors=True)
            removed.append(page.page_number)
    return removed

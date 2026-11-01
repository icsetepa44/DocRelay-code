from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData, PageExtraction
from src.storage import run_store

_REQUIRED_PAGE_FILES = ("extraction.json", "carry_state.json")


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def page_complete(page_dir: Path) -> bool:
    return all(_load_json(page_dir / name) is not None for name in _REQUIRED_PAGE_FILES)


def _stitch_seq(stitch_id_value: Optional[str]) -> int:
    if not stitch_id_value:
        return 0
    tail = stitch_id_value.rsplit(":", 1)[-1]
    if not tail.startswith("s") or not tail[1:].isdigit():
        return 0
    return int(tail[1:])


def _max_stitch_seq(extractions: List[PageExtraction], carry: CarryState) -> int:
    best = 0
    for pe in extractions:
        for b in pe.blocks:
            best = max(best, _stitch_seq(b.stitch_id))
    for u in carry.open_units:
        best = max(best, _stitch_seq(u.stitch_id))
    return best


def _reconstruct_summary(page_dir: Path, pe: PageExtraction, carry: CarryState,
                         page: PageData, min_chars_nontrivial: int) -> Dict[str, Any]:
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
    index: int
    extractions: List[PageExtraction] = field(default_factory=list)
    carry: CarryState = field(default_factory=CarryState.empty)
    allocator_start: int = 0
    page_summaries: List[Dict[str, Any]] = field(default_factory=list)
    resume_page_number: Optional[int] = None


def compute_resume_state(run_dir: Path, doc_name: str, model_label: Optional[str],
                         pages: List[PageData], *, min_chars_nontrivial: int) -> ResumeState:
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
    removed: List[int] = []
    for page in pages[index:]:
        pdir = run_store.page_dir_path(run_dir, doc_name, page.page_number, model_label)
        if pdir.exists():
            shutil.rmtree(pdir, ignore_errors=True)
            removed.append(page.page_number)
    return removed

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.stages._common import block_text
from src.stages.carry_rejection import MAINRULE, PageBlock

DEFAULT_MIN_RUN = 4
DEFAULT_MAX_ITEM_CHARS = 160


def _qualifies(b: PageBlock, referenced_parents: set, max_item_chars: int) -> bool:
    block_id, block_type, parent_block_id, stitch_id, text = b
    text = text or ""
    return (block_type == "text_block"
            and parent_block_id is None
            and stitch_id is None
            and block_id not in referenced_parents
            and bool(MAINRULE.match(text))
            and len(text.strip()) <= max_item_chars)


def detect_uniform_numbered_runs(
    page_blocks: Sequence[PageBlock],
    min_run: int = DEFAULT_MIN_RUN,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
) -> List[Dict[str, Any]]:
    referenced_parents = {b[2] for b in page_blocks if b[2]}
    quals = [_qualifies(b, referenced_parents, max_item_chars) for b in page_blocks]

    runs: List[Dict[str, Any]] = []
    i, n = 0, len(page_blocks)
    while i < n:
        if not quals[i]:
            i += 1
            continue
        j = i
        while j < n and quals[j]:
            j += 1
        if j - i >= min_run:
            run_blocks = list(page_blocks[i:j])
            owner_id, kind = _owning_parent(page_blocks, i)
            runs.append({
                "block_ids": [b[0] for b in run_blocks],
                "owning_parent_id": owner_id,
                "parent_kind": kind,
            })
        i = j
    return runs


def _owning_parent(page_blocks: Sequence[PageBlock], run_start: int) -> tuple[str, str]:
    for k in range(run_start - 1, -1, -1):
        if page_blocks[k][1] == "heading":
            return page_blocks[k][0], "schedule_heading"
    return page_blocks[run_start][0], "self"


def analyze_over_segmentation(
    page_blocks: Sequence[PageBlock],
    min_run: int = DEFAULT_MIN_RUN,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
) -> Dict[str, Any]:
    runs = detect_uniform_numbered_runs(page_blocks, min_run, max_item_chars)
    return {
        "runs": runs,
        "run_count": len(runs),
        "flagged_block_count": sum(len(r["block_ids"]) for r in runs),
        "min_run": min_run,
        "max_item_chars": max_item_chars,
    }


def collapse_uniform_runs(blocks: List[Any], runs: Sequence[Dict[str, Any]]) -> List[Any]:
    by_id = {b.block_id: b for b in blocks}
    for run in runs:
        owner_id = run["owning_parent_id"]
        for bid in run["block_ids"]:
            if bid == owner_id:
                continue
            blk = by_id.get(bid)
            if blk is not None:
                blk.parent_block_id = owner_id
    return blocks


def page_blocks_of(blocks: Sequence[Any]) -> List[PageBlock]:
    return [(b.block_id, b.block_type, b.parent_block_id, b.stitch_id, block_text(b)) for b in blocks]

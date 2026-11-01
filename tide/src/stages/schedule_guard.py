"""
Schedule over-segmentation guard (T1 / T4 safety net) — deterministic, no LLM.

When the extractor shreds a uniform serial/appendix catalogue (a schedule that reads as a long,
parallel-structured numbered list) into many separate top-level numbered ``text_block``s, this
module can (a) DETECT that run for the over-segmentation ledger, and (b) COLLAPSE it by RE-NESTING
— never by fabricating a table. Detection is representation-agnostic (the same
``(block_id, block_type, parent_block_id, stitch_id, text)`` tuple shape ``carry_rejection`` uses),
so it works in both ``page`` and ``region`` retrieval mode and needs no region-type field on a
block (there isn't one).

A run qualifies only when every member is a SHORT, top-level, null-stitch, childless ``text_block``
that opens with a main-rule numeral (``carry_rejection.MAINRULE``). The short-item gate excludes
genuine operative rules, which carry real bodies. This is a STRUCTURAL signal — not a keyword match
on content — consistent with the project's no-regex-semantic-extraction rule.

The detector is always-on (diagnostic only). The collapse is opt-in (``config.SCHEDULE_COLLAPSE``);
when it runs it only rewrites ``parent_block_id`` — every block's text is preserved byte-for-byte.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.stages._common import block_text
from src.stages.carry_rejection import MAINRULE, PageBlock

# Defaults; the orchestrator passes the live config values.
DEFAULT_MIN_RUN = 4
DEFAULT_MAX_ITEM_CHARS = 160


def _qualifies(b: PageBlock, referenced_parents: set, max_item_chars: int) -> bool:
    """A single block that could be one shredded catalogue item: a short, top-level, null-stitch,
    childless ``text_block`` opening with a main-rule numeral."""
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
    """Find maximal runs of >= ``min_run`` consecutive qualifying blocks (see ``_qualifies``).

    Returns one dict per run: ``{block_ids, owning_parent_id, parent_kind}`` where the owner is the
    nearest preceding ``heading`` on the page (``parent_kind="schedule_heading"``) if any, else the
    run's own first block (``parent_kind="self"``). The owner is what a collapse would re-nest the
    run under; it is recorded even when collapse is disabled so the diagnostic is actionable.
    """
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
    """The block a run should hang off: the nearest preceding ``heading`` (a schedule/appendix
    cover), else the run's own first block."""
    for k in range(run_start - 1, -1, -1):
        if page_blocks[k][1] == "heading":
            return page_blocks[k][0], "schedule_heading"
    return page_blocks[run_start][0], "self"


def analyze_over_segmentation(
    page_blocks: Sequence[PageBlock],
    min_run: int = DEFAULT_MIN_RUN,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
) -> Dict[str, Any]:
    """Diagnostic record for one page (persisted as ``over_segmentation.json``). Never mutates."""
    runs = detect_uniform_numbered_runs(page_blocks, min_run, max_item_chars)
    return {
        "runs": runs,
        "run_count": len(runs),
        "flagged_block_count": sum(len(r["block_ids"]) for r in runs),
        "min_run": min_run,
        "max_item_chars": max_item_chars,
    }


def collapse_uniform_runs(blocks: List[Any], runs: Sequence[Dict[str, Any]]) -> List[Any]:
    """Re-nest each detected run under its owning parent, in place (T1/T4 collapse).

    Only ``parent_block_id`` is rewritten; every block's ``content``/text is untouched. For a
    ``schedule_heading`` owner all run blocks become children of the cover heading; for a ``self``
    owner the run's first block stays top-level and its siblings become its children. Blocks not in
    a run are left exactly as-is. Returns the same list (mutated) for call-site clarity.
    """
    by_id = {b.block_id: b for b in blocks}
    for run in runs:
        owner_id = run["owning_parent_id"]
        for bid in run["block_ids"]:
            if bid == owner_id:
                continue  # the run's own first block (self owner) stays top-level
            blk = by_id.get(bid)
            if blk is not None:
                blk.parent_block_id = owner_id
    return blocks


def page_blocks_of(blocks: Sequence[Any]) -> List[PageBlock]:
    """Build the ``(block_id, block_type, parent_block_id, stitch_id, text)`` tuples the detector
    consumes from a list of ``Block`` objects (reuses the shared ``block_text`` idiom)."""
    return [(b.block_id, b.block_type, b.parent_block_id, b.stitch_id, block_text(b)) for b in blocks]

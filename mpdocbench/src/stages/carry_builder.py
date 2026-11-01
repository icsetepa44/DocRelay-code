"""
Carry Builder (spec §8.7, §23) — deterministic, no LLM.

Emits the CarryState (open_units) handed to the next page. For every block the extractor
confirmed "continues": reuse its stitch_id if it already has one (continuing from a
previous page), otherwise allocate a new one and set it on both the block and the
continuation decision.

B1 (open-main-point model): open units are keyed by stitch_id, so several continuing
fragments of the SAME unit on one page fold into ONE OpenUnit instead of spawning a slot
each. A MAIN point's is_main_point / expected_next_main_point / continuation_hint persist
from the carried (incoming) unit across pages where only its sub-points stream, so a main
point stays open until the extractor stops marking it "continues" — i.e. until the next
main point of equal/higher rank (or the document end) actually appears. A unit that ends
on this page is simply not carried forward.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.contracts.continuation import CarryState, ContinuationDecision, OpenUnit
from src.contracts.page import PageExtraction
from src.core import config
from src.core.ids import StitchIdAllocator
from src.stages._common import block_text


def _short_state(block) -> str:
    # Source via block_text so a table block yields its flattened cells rather than "" —
    # otherwise an open table's state degrades to the boilerplate below (T6).
    text = block_text(block).replace("\n", " ").strip()
    if text:
        return (text[:200] + "…") if len(text) > 200 else text
    return f"An open {block.block_type} continues onto the next page."


def _accumulate(existing: str, fragment: str) -> str:
    """Append a new continuing fragment to the text carried from earlier pages."""
    existing = (existing or "").strip()
    fragment = (fragment or "").strip()
    if not existing:
        return fragment
    if not fragment or fragment in existing:
        return existing
    return existing + "\n" + fragment


def _bounded(text: str, cap: int) -> str:
    """Bound carried text to `cap` chars, keeping the head (identifies the main point)
    and the tail (where it resumes) with an elision marker in between. cap<=0 -> ""."""
    text = text or ""
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    budget = max(0, cap - 40)          # leave room for the elision marker
    head = budget // 3
    tail = budget - head
    omitted = len(text) - head - tail
    marker = f"…[omitted {omitted} chars]…"
    return text[:head] + marker + (text[len(text) - tail:] if tail else "")


def _open_text(base_text: str, block, cap: int) -> str:
    """Accumulated, bounded open-unit text for the carry. "" when enrichment is disabled."""
    if cap <= 0:
        return ""
    return _bounded(_accumulate(base_text, block_text(block)), cap)


def build_carry_state(
    extraction: PageExtraction,
    final_continuation: ContinuationDecision,
    allocator: StitchIdAllocator,
    incoming_carry: Optional[CarryState] = None,
) -> CarryState:
    dec_by_block = {d.block_id: d for d in final_continuation.decisions}
    incoming = {u.stitch_id: u for u in (incoming_carry.open_units if incoming_carry else [])}

    units_by_stitch: Dict[str, OpenUnit] = {}
    order: List[str] = []

    for block in extraction.blocks:
        decision = dec_by_block.get(block.block_id)
        if decision is None or decision.decision != "continues":
            continue

        if not block.stitch_id:
            block.stitch_id = allocator.next()
        decision.stitch_id = block.stitch_id
        sid = block.stitch_id
        base = incoming.get(sid)  # the same unit, already open from a previous page

        if sid in units_by_stitch:
            # A later continuing fragment of the SAME unit on this page: fold it in,
            # do NOT open a second slot for it.
            u = units_by_stitch[sid]
            u.last_block_id = block.block_id
            u.state = _short_state(block)
            u.open_text = _open_text(u.open_text, block, config.CARRY_OPEN_TEXT_MAX_CHARS)
            if decision.evidence:
                u.continuation_evidence = list(decision.evidence)
            continue

        order.append(sid)
        units_by_stitch[sid] = OpenUnit(
            stitch_id=sid,
            last_block_id=block.block_id,
            block_type=(base.block_type if base else block.block_type),
            state=_short_state(block),
            open_text=_open_text(
                (base.open_text if base else ""), block, config.CARRY_OPEN_TEXT_MAX_CHARS),
            continuation_evidence=list(decision.evidence),
            skills=(list(block.skills_used) or (list(base.skills) if base else [])),
        )

    return CarryState(
        from_page=extraction.page_number,
        open_units=[units_by_stitch[s] for s in order],
    )

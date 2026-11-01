"""
Assembler (spec §8.14, §28) — deterministic, no LLM.

Builds FinalDocument logical units from accepted page extractions:
  - stitch_id == null  -> one unit (unit_id = block_id), content = the block's content.
  - stitch_id set       -> all fragments sharing that stitch_id merge into ONE unit
                           (unit_id = stitch_id) whose content is the ORDERED list of
                           {block_type, content} fragments (never flattened).
Parent-child hierarchy (parent_block_id) is preserved via parent_unit_id references.

B3: a merged unit's fragments AND the units themselves are ordered by GLOBAL reading
order — (page, sequence) parsed from each source block_id — instead of trusting input
iteration order; and an exact token-run overlap at a page boundary (the same trailing
text repeated at the head of the next fragment) is removed conservatively.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.contracts.final_document import FinalDocument, FinalUnit
from src.contracts.page import PageExtraction
from src.core.ids import unit_id_for

# B3: only strip a boundary overlap of at least this many EXACT consecutive tokens
# (short coincidental repeats are left alone); cap the compared window for cost.
BOUNDARY_OVERLAP_MIN_TOKENS = 4
BOUNDARY_OVERLAP_MAX_TOKENS = 60


def _page_seq(block_id: str) -> tuple[int, int]:
    """(page, sequence) parsed from a canonical block_id `{doc}:pXXXX:bYYYY`.

    rsplit from the right tolerates a document_id that itself contains ':'.
    """
    try:
        parts = block_id.rsplit(":", 2)
        return (int(parts[-2][1:]), int(parts[-1][1:]))
    except (IndexError, ValueError):
        return (0, 0)


def remove_boundary_overlap(
    prev_text: str,
    next_text: str,
    min_tokens: int = BOUNDARY_OVERLAP_MIN_TOKENS,
    max_tokens: int = BOUNDARY_OVERLAP_MAX_TOKENS,
) -> str:
    """Remove from the head of next_text an EXACT suffix/prefix token-run that prev_text
    already contains. Removes only a genuine duplicate (>= min_tokens exact consecutive
    tokens); never invents or deletes non-duplicate content — the kept remainder is
    character-exact.
    """
    if not prev_text or not next_text:
        return next_text
    prev_tokens = prev_text.split()
    next_tokens = next_text.split()
    if not prev_tokens or not next_tokens:
        return next_text
    limit = min(len(prev_tokens), len(next_tokens), max_tokens)
    best_k = 0
    for k in range(limit, min_tokens - 1, -1):  # largest exact run wins
        if prev_tokens[-k:] == next_tokens[:k]:
            best_k = k
            break
    if best_k == 0:
        return next_text
    # Drop exactly best_k leading whitespace-delimited tokens; keep the rest char-exact.
    idx, count, n = 0, 0, len(next_text)
    while idx < n and count < best_k:
        while idx < n and next_text[idx].isspace():
            idx += 1
        while idx < n and not next_text[idx].isspace():
            idx += 1
        count += 1
    return next_text[idx:].lstrip()


def assemble(document_id: str, pages: List[PageExtraction]) -> FinalDocument:
    # global block_id -> unit_id
    unit_of: Dict[str, str] = {}
    for pe in pages:
        for b in pe.blocks:
            unit_of[b.block_id] = unit_id_for(b.block_id, b.stitch_id)

    # Gather every block per unit; remember each unit's earliest (page, seq) and parent.
    frags: Dict[str, List[Any]] = {}
    first_seen: Dict[str, tuple[int, int]] = {}
    parent_of_unit: Dict[str, Optional[str]] = {}
    for pe in pages:
        for b in pe.blocks:
            uid = unit_of[b.block_id]
            frags.setdefault(uid, []).append(b)
            ps = _page_seq(b.block_id)
            if uid not in first_seen or ps < first_seen[uid]:
                first_seen[uid] = ps
            if uid not in parent_of_unit:
                p = unit_of.get(b.parent_block_id) if b.parent_block_id else None
                parent_of_unit[uid] = (None if p == uid else p)

    units_by_id: Dict[str, FinalUnit] = {}
    for uid, blist in frags.items():
        blist.sort(key=lambda b: _page_seq(b.block_id))  # B3: fragment reading order
        head = blist[0]
        stitched = bool(head.stitch_id)
        if not stitched and len(blist) == 1:
            content: Any = head.content
        else:
            content = []
            prev_text = ""
            for b in blist:
                frag = {"block_type": b.block_type, "content": dict(b.content or {})}
                key = ("text" if "text" in frag["content"]
                       else ("description" if "description" in frag["content"] else None))
                if key and prev_text:  # B3: conservative boundary-overlap dedupe
                    frag["content"][key] = remove_boundary_overlap(
                        prev_text, str(frag["content"][key]))
                if key:
                    prev_text = str(frag["content"][key])
                content.append(frag)
        units_by_id[uid] = FinalUnit(
            unit_id=uid,
            block_type=head.block_type,
            parent_unit_id=parent_of_unit.get(uid),
            source_block_ids=[b.block_id for b in blist],
            content=content,
        )

    # B3: emit units in global reading order (earliest source block first).
    order = sorted(units_by_id.keys(), key=lambda u: first_seen[u])
    return FinalDocument(document_id=document_id, units=[units_by_id[u] for u in order])

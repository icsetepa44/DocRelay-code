"""
Carry-state rejection analysis (over-segmentation diagnostic) — deterministic, no LLM.

An incoming carry-state hands the next page a set of OPEN units (any block type: a
numbered point, a spanning table, a schedule/appendix cover heading, a long paragraph).
A block on page N "continues" an open unit S by REUSING its stitch_id (see
`prompts/extractor.md` rule 5 and `carry_builder.build_carry_state`, which opens a unit for
ANY block whose continuation decision is "continues"). So the general, type-agnostic signal
is: did any block on page N carry `stitch_id == S`?

  continued           — some block on page N reused S (the extractor honoured the carry).
  closed_by_successor — a MAIN point was dropped, but a fresh main number begins on page N
                        (legitimate close: the point ended and the next one opened).
  rejected            — S was dropped with no legitimate close; the extractor should have
                        continued it. Sub-typed by `symptom` (below).
  dropped_ambiguous   — a non-main paragraph dropped with no symptom; recorded, not flagged
                        (a plain paragraph can legitimately end without a marker).

symptoms (why a drop looks like a rejection):
  promoted_subpoint   — a bracketed sub-point became a top-level, null-stitch block
                        (the classic over-segmentation: sub-points of a dropped main point).
  restarted_table     — a fresh top-level, null-stitch `table` appeared (a spanning table
                        was restarted instead of continued).
  table_dropped       — an open `table` vanished with no restart in view.
  schedule_dropped    — an open `heading` (schedule/appendix cover) vanished.
  main_dropped        — an open main point vanished with no successor and no promotion.

This is a DIAGNOSTIC: it records a status for every open unit and hard-flags only the
confident rejections, so results can be filtered by `symptom` / `block_type` to find
patterns. Representation-agnostic: callers pass plain dicts/tuples (works for both the live
orchestrator, on `Block`/`OpenUnit` objects, and the post-hoc detector, on JSON dicts).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Leading bracketed sub-point marker: (ক) (খ) (১) (a) (i) [ক] …
SUBPOINT = re.compile(r"^\s*[\(\[][^\)\]]{1,4}[\)\]।]")
# Leading main-rule numeral then danda/dot: ৩।  12.  ১২।
MAINRULE = re.compile(r"^\s*(?:[০-৯]+|\d+)\s*[।.]")
# Digits that lead a main-rule label (Bangla ০-৯ or ASCII), for ordinal comparison.
_LEADING_DIGITS = re.compile(r"\s*([০-৯\d]+)")
_BN_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _leading_ordinal(text: Optional[str]) -> Optional[int]:
    """Parse the leading main-rule ordinal of a label/line ('১০। …' -> 10, '3.' -> 3).

    Gated on MAINRULE, so a bare number without a danda/dot (e.g. a gazette year) is ignored.
    Normalizes Bangla ০-৯ and ASCII digits. Returns None when there is no leading main-rule
    numeral, so callers can fall back rather than guess. Language-agnostic numeric primitive."""
    if not text or not MAINRULE.match(text):
        return None
    m = _LEADING_DIGITS.match(text)
    if not m:
        return None
    try:
        return int(m.group(1).translate(_BN_TO_ASCII))
    except ValueError:
        return None


def is_plausible_successor(expected_label: Optional[str], fresh_text: Optional[str]) -> bool:
    """Is `fresh_text` a plausible NEXT main point after an open unit expecting `expected_label`?

    The carry collision guard (T11) uses this: a stray shredded form-field like '২। জাতীয়তা'
    matches MAINRULE but is NOT the successor that a rule-10 open unit is waiting for, so it must
    not be treated as 'the next main number is visible' and close the real open rule. True when
    the fresh ordinal is >= the expected ordinal; True (fall back — never over-suppress a genuine
    close) when either label is missing or unparseable."""
    exp = _leading_ordinal(expected_label)
    got = _leading_ordinal(fresh_text)
    if exp is None or got is None:
        return True
    return got >= exp


CLASS_MEANING = {
    "continued": "the extractor reused the open unit's stitch_id (honoured the carry)",
    "closed_by_successor": "main point dropped but a fresh main number opened (legit close)",
    "rejected": "open unit dropped with no legitimate close (the extractor should have continued it)",
    "dropped_ambiguous": "non-main paragraph dropped with no symptom (recorded, not flagged)",
}

# A page-block is (block_id, block_type, parent_block_id, stitch_id, text).
PageBlock = Tuple[str, str, Optional[str], Optional[str], str]


def is_promoted_subpoint(block_type: str, parent_block_id: Optional[str],
                         stitch_id: Optional[str], text: str) -> bool:
    """A top-level text_block that begins with a sub-point marker and carries no stitch_id —
    i.e. the assembler will NOT merge it, so it becomes a standalone clause."""
    return (block_type == "text_block" and parent_block_id is None
            and stitch_id is None and bool(SUBPOINT.match(text or "")))


def is_fresh_main(block_type: str, parent_block_id: Optional[str],
                  stitch_id: Optional[str], text: str) -> bool:
    """A fresh (null-stitch) top-level block whose text opens a new main-numbered point —
    'the next main number is visible on this page'."""
    return (block_type == "text_block" and parent_block_id is None
            and stitch_id is None and bool(MAINRULE.match(text or "")))


def is_fresh_table(block_type: str, parent_block_id: Optional[str],
                   stitch_id: Optional[str], text: str) -> bool:
    """A fresh (null-stitch) top-level table — a restarted table rather than a continuation."""
    return block_type == "table" and parent_block_id is None and stitch_id is None


def _classify(unit: Dict[str, Any], reused: bool, *, has_fresh_main: bool,
              has_promoted_subpoint: bool, has_fresh_table: bool) -> Tuple[str, Optional[str]]:
    """Return (status, symptom) for one open unit given the page's reuse + symptom signals."""
    if reused:
        return "continued", None
    block_type = unit.get("block_type")
    if unit.get("is_main_point"):
        if has_fresh_main:
            return "closed_by_successor", None
        return "rejected", ("promoted_subpoint" if has_promoted_subpoint else "main_dropped")
    if block_type == "table":
        return "rejected", ("restarted_table" if has_fresh_table else "table_dropped")
    if block_type == "heading":
        return "rejected", "schedule_dropped"
    # Non-main text (a sub-point/paragraph fragment): only escalate on a clear symptom.
    if has_promoted_subpoint:
        return "rejected", "promoted_subpoint"
    return "dropped_ambiguous", None


def analyze_carry(incoming_units: Sequence[Dict[str, Any]],
                  page_blocks: Sequence[PageBlock],
                  error_adjacent: bool = False,
                  collision_guard: bool = False) -> Dict[str, Any]:
    """Classify each incoming open unit as continued / closed / rejected on this page.

    incoming_units : dicts with stitch_id, block_type, is_main_point, and (optional)
                     state, open_text, expected_next_main_point.
    page_blocks    : (block_id, block_type, parent_block_id, stitch_id, text) in reading order.
    error_adjacent : whether this page (or the one before) hit a pipeline error — annotated
                     on every unit so a consumer can down-weight error-driven drops.
    collision_guard: T11 guard — when True, an open MAIN point is only 'closed_by_successor' if a
                     fresh main on the page is a plausible successor of its expected_next_main_point
                     (mirrors the live orchestrator guard so the ledger agrees). Default False keeps
                     the historical 'any fresh main closes it' behaviour byte-for-byte.
    Returns        : {"units": [ {…unit…, status, symptom, error_adjacent} ], "rejected_count": N}.
    """
    reused_ids = {b[3] for b in page_blocks if b[3]}
    fresh_main_texts = [b[4] for b in page_blocks if is_fresh_main(b[1], b[2], b[3], b[4])]
    has_fresh_main = bool(fresh_main_texts)
    has_promoted_subpoint = any(is_promoted_subpoint(b[1], b[2], b[3], b[4]) for b in page_blocks)
    has_fresh_table = any(is_fresh_table(b[1], b[2], b[3], b[4]) for b in page_blocks)

    units: List[Dict[str, Any]] = []
    for u in incoming_units:
        # Collision guard (T11): for an open MAIN point, only treat a fresh main as its successor
        # when the fresh ordinal plausibly follows this unit's expected_next_main_point — so a
        # stray shredded field-number can't masquerade as the next rule and close a real one.
        if collision_guard and has_fresh_main and u.get("is_main_point"):
            unit_has_fresh_main = any(
                is_plausible_successor(u.get("expected_next_main_point"), t)
                for t in fresh_main_texts)
        else:
            unit_has_fresh_main = has_fresh_main
        status, symptom = _classify(
            u, u.get("stitch_id") in reused_ids,
            has_fresh_main=unit_has_fresh_main,
            has_promoted_subpoint=has_promoted_subpoint,
            has_fresh_table=has_fresh_table,
        )
        units.append({
            "stitch_id": u.get("stitch_id"),
            "block_type": u.get("block_type"),
            "is_main_point": bool(u.get("is_main_point")),
            "expected_next_main_point": u.get("expected_next_main_point"),
            "state": u.get("state"),
            "open_text": u.get("open_text"),
            "status": status,
            "symptom": symptom,
            "error_adjacent": error_adjacent,
        })
    return {"units": units, "rejected_count": sum(1 for r in units if r["status"] == "rejected")}

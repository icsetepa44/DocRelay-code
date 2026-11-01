from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUBPOINT = re.compile(r"^\s*[\(\[][^\)\]]{1,4}[\)\]।]")
MAINRULE = re.compile(r"^\s*(?:[০-৯]+|\d+)\s*[।.]")
_LEADING_DIGITS = re.compile(r"\s*([০-৯\d]+)")
_BN_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _leading_ordinal(text: Optional[str]) -> Optional[int]:
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

PageBlock = Tuple[str, str, Optional[str], Optional[str], str]


def is_promoted_subpoint(block_type: str, parent_block_id: Optional[str],
                         stitch_id: Optional[str], text: str) -> bool:
    return (block_type == "text_block" and parent_block_id is None
            and stitch_id is None and bool(SUBPOINT.match(text or "")))


def is_fresh_main(block_type: str, parent_block_id: Optional[str],
                  stitch_id: Optional[str], text: str) -> bool:
    return (block_type == "text_block" and parent_block_id is None
            and stitch_id is None and bool(MAINRULE.match(text or "")))


def is_fresh_table(block_type: str, parent_block_id: Optional[str],
                   stitch_id: Optional[str], text: str) -> bool:
    return block_type == "table" and parent_block_id is None and stitch_id is None


def _classify(unit: Dict[str, Any], reused: bool, *, has_fresh_main: bool,
              has_promoted_subpoint: bool, has_fresh_table: bool) -> Tuple[str, Optional[str]]:
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
    if has_promoted_subpoint:
        return "rejected", "promoted_subpoint"
    return "dropped_ambiguous", None


def analyze_carry(incoming_units: Sequence[Dict[str, Any]],
                  page_blocks: Sequence[PageBlock],
                  error_adjacent: bool = False,
                  collision_guard: bool = False) -> Dict[str, Any]:
    reused_ids = {b[3] for b in page_blocks if b[3]}
    fresh_main_texts = [b[4] for b in page_blocks if is_fresh_main(b[1], b[2], b[3], b[4])]
    has_fresh_main = bool(fresh_main_texts)
    has_promoted_subpoint = any(is_promoted_subpoint(b[1], b[2], b[3], b[4]) for b in page_blocks)
    has_fresh_table = any(is_fresh_table(b[1], b[2], b[3], b[4]) for b in page_blocks)

    units: List[Dict[str, Any]] = []
    for u in incoming_units:
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

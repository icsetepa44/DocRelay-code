"""ContinuationDecision (spec §4.7) and CarryState (spec §4.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import SkillReference, to_dict


@dataclass
class BlockContinuation:
    """One block's continuation decision. decision ∈ {continues, ends, uncertain}.

    B1 fields (all optional, default-absent so older data / weak models still parse):
      is_main_point            — this block is (a fragment of) a top-level MAIN numbered point.
      expected_next_main_point — label expected to open the NEXT main point (e.g. "৩।", "3.").
      continuation_hint        — short note on what remains open under this unit.
    """

    block_id: str
    stitch_id: Optional[str]
    decision: str
    evidence: List[str] = field(default_factory=list)
    is_main_point: Optional[bool] = None
    expected_next_main_point: Optional[str] = None
    continuation_hint: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BlockContinuation":
        return BlockContinuation(
            block_id=str(d["block_id"]),
            stitch_id=(str(d["stitch_id"]) if d.get("stitch_id") else None),
            decision=str(d.get("decision", "uncertain")),
            evidence=[str(e) for e in (d.get("evidence") or [])],
            is_main_point=(bool(d["is_main_point"]) if d.get("is_main_point") is not None else None),
            expected_next_main_point=(
                str(d["expected_next_main_point"]) if d.get("expected_next_main_point") else None),
            continuation_hint=(str(d["continuation_hint"]) if d.get("continuation_hint") else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = to_dict(self)
        # Omit B1 signals when unset so pre-B1 data round-trips byte-identically.
        for k in ("is_main_point", "expected_next_main_point", "continuation_hint"):
            if d.get(k) is None:
                d.pop(k, None)
        return d


@dataclass
class ContinuationDecision:
    page_number: int
    decisions: List[BlockContinuation] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ContinuationDecision":
        return ContinuationDecision(
            page_number=int(d.get("page_number", 0)),
            decisions=[BlockContinuation.from_dict(x) for x in (d.get("decisions") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "decisions": [x.to_dict() for x in self.decisions],
        }


@dataclass
class OpenUnit:
    """A logical unit still open across a page boundary (spec §4.5)."""

    stitch_id: str
    last_block_id: str
    block_type: str
    state: str
    # Actual text of the still-open unit carried across the page boundary (head+tail,
    # bounded by config.CARRY_OPEN_TEXT_MAX_CHARS). "" when enrichment is disabled.
    open_text: str = ""
    continuation_evidence: List[str] = field(default_factory=list)
    skills: List[SkillReference] = field(default_factory=list)
    # B1: a main point stays open (across pages) until the next main point of equal/
    # higher rank appears; these signals persist while only its sub-points stream.
    is_main_point: bool = False
    expected_next_main_point: Optional[str] = None
    continuation_hint: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpenUnit":
        return OpenUnit(
            stitch_id=str(d["stitch_id"]),
            last_block_id=str(d["last_block_id"]),
            block_type=str(d.get("block_type", "text_block")),
            state=str(d.get("state", "")),
            open_text=str(d.get("open_text", "")),
            continuation_evidence=[str(e) for e in (d.get("continuation_evidence") or [])],
            skills=SkillReference.list_from(d.get("skills")),
            is_main_point=bool(d.get("is_main_point", False)),
            expected_next_main_point=(
                str(d["expected_next_main_point"]) if d.get("expected_next_main_point") else None),
            continuation_hint=(str(d["continuation_hint"]) if d.get("continuation_hint") else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = to_dict(self)
        # Omit B1 signals at default so pre-B1 carry state round-trips byte-identically.
        if not d.get("is_main_point"):
            d.pop("is_main_point", None)
        for k in ("expected_next_main_point", "continuation_hint"):
            if d.get(k) is None:
                d.pop(k, None)
        # Omit open_text when empty so pre-enrichment carry state round-trips byte-identically.
        if not d.get("open_text"):
            d.pop("open_text", None)
        return d


@dataclass
class CarryState:
    """What remains open going into the next page. Compact — no accumulated content."""

    from_page: int
    open_units: List[OpenUnit] = field(default_factory=list)

    @staticmethod
    def empty(from_page: int = -1) -> "CarryState":
        return CarryState(from_page=from_page, open_units=[])

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CarryState":
        return CarryState(
            from_page=int(d.get("from_page", -1)),
            open_units=[OpenUnit.from_dict(u) for u in (d.get("open_units") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_page": self.from_page,
            "open_units": [u.to_dict() for u in self.open_units],
        }

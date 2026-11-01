"""FinalDocument / FinalUnit (spec §4.15) — logical units, not page fragments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FinalUnit:
    """
    unit_id = stitch_id if the unit spans/participates in continuation, else block_id.

    content is either:
      - a single content dict (for a standalone, non-stitched block), or
      - an ordered list of {block_type, content} children (for a stitched unit) — the
        text/table/text ordering is preserved, never flattened (spec §28).
    """

    unit_id: str
    block_type: str
    parent_unit_id: Optional[str]
    source_block_ids: List[str] = field(default_factory=list)
    content: Union[Dict[str, Any], List[Dict[str, Any]]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FinalUnit":
        return FinalUnit(
            unit_id=str(d["unit_id"]),
            block_type=str(d["block_type"]),
            parent_unit_id=(str(d["parent_unit_id"]) if d.get("parent_unit_id") else None),
            source_block_ids=[str(b) for b in (d.get("source_block_ids") or [])],
            content=d.get("content", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "block_type": self.block_type,
            "parent_unit_id": self.parent_unit_id,
            "source_block_ids": list(self.source_block_ids),
            "content": self.content,
        }


@dataclass
class FinalDocument:
    document_id: str
    units: List[FinalUnit] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FinalDocument":
        return FinalDocument(
            document_id=str(d["document_id"]),
            units=[FinalUnit.from_dict(u) for u in (d.get("units") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "units": [u.to_dict() for u in self.units],
        }

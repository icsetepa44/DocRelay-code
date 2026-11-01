"""
Block — the canonical extracted page-level block (spec §4.6).

Every block always contains: block_id, block_type, parent_block_id, stitch_id,
content, skills_used. `content` shape varies by block_type (text/heading/table/…),
so it is kept as a free-form dict here; block-type-specific validation lives in the
JSON schemas and the extractor normalizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import SkillReference, to_dict


@dataclass
class Block:
    block_id: str
    block_type: str
    parent_block_id: Optional[str]
    stitch_id: Optional[str]
    content: Dict[str, Any] = field(default_factory=dict)
    skills_used: List[SkillReference] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Block":
        return Block(
            block_id=str(d["block_id"]),
            block_type=str(d["block_type"]),
            parent_block_id=(str(d["parent_block_id"]) if d.get("parent_block_id") else None),
            stitch_id=(str(d["stitch_id"]) if d.get("stitch_id") else None),
            content=dict(d.get("content") or {}),
            skills_used=SkillReference.list_from(d.get("skills_used")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


COVERAGE_VALUES = ("sufficient", "partial", "insufficient")
CONTINUATION_DECISIONS = ("continues", "ends", "uncertain")
SKILL_STATUSES = ("main", "fallback", "candidate", "deprecated")
VALIDATION_STATUSES = ("pass", "fail")
PAGE_ISSUE_TYPES = (
    "missing_content",
    "duplicated_content",
    "wrong_reading_order",
    "wrong_block_type",
    "wrong_relationship",
    "broken_hierarchy",
    "table_structure_error",
    "continuation_error",
    "unrelated_content_merged",
    "logical_unit_split",
    "other",
)


def to_dict(obj: Any) -> Any:
    return asdict(obj)


@dataclass(frozen=True)
class SkillReference:
    skill_id: str
    version: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SkillReference":
        return SkillReference(skill_id=str(d["skill_id"]), version=int(d["version"]))

    @staticmethod
    def list_from(items: Optional[List[Dict[str, Any]]]) -> List["SkillReference"]:
        out: List[SkillReference] = []
        for it in items or []:
            if it is None:
                continue
            try:
                out.append(SkillReference.from_dict(it))
            except (KeyError, TypeError, ValueError):
                continue
        return out

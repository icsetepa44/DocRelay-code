from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import to_dict
from .continuation import BlockContinuation, ContinuationDecision


@dataclass
class SuperValidationIssue:
    issue_type: str
    description: str
    target_block_id: Optional[str] = None
    suggested_fix: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> SuperValidationIssue:
        return SuperValidationIssue(
            issue_type=str(d.get("issue_type", "other")),
            description=str(d.get("description", "")),
            target_block_id=d.get("target_block_id"),
            suggested_fix=d.get("suggested_fix"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class SuperValidationResult:
    page_number: int
    status: str = "clean"
    issues: List[SuperValidationIssue] = field(default_factory=list)
    continuation_review: List[BlockContinuation] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return self.status == "has_issues" or len(self.issues) > 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> SuperValidationResult:
        return SuperValidationResult(
            page_number=int(d.get("page_number", 0)),
            status=str(d.get("status", "clean")),
            issues=[SuperValidationIssue.from_dict(i) for i in (d.get("issues") or [])],
            continuation_review=[
                BlockContinuation(
                    block_id=str(c.get("block_id", "")),
                    decision=str(c.get("decision", "ends")),
                    stitch_id=c.get("stitch_id"),
                    is_main_point=c.get("is_main_point"),
                    expected_next_main_point=c.get("expected_next_main_point"),
                    evidence=list(c.get("evidence") or []),
                )
                for c in (d.get("continuation_review") or [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "continuation_review": [c.to_dict() for c in self.continuation_review],
        }

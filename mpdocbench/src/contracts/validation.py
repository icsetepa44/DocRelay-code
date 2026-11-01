from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .common import to_dict


@dataclass
class ValidationIssue:
    block_ids: List[str] = field(default_factory=list)
    type: str = "other"
    description: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ValidationIssue":
        return ValidationIssue(
            block_ids=[str(b) for b in (d.get("block_ids") or [])],
            type=str(d.get("type", "other")),
            description=str(d.get("description", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class PageValidation:
    page_number: int
    status: str = "pass"
    issues: List[ValidationIssue] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PageValidation":
        return PageValidation(
            page_number=int(d.get("page_number", 0)),
            status=str(d.get("status", "pass")),
            issues=[ValidationIssue.from_dict(i) for i in (d.get("issues") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class FinalValidationIssue:
    unit_ids: List[str] = field(default_factory=list)
    type: str = "other"
    description: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FinalValidationIssue":
        return FinalValidationIssue(
            unit_ids=[str(u) for u in (d.get("unit_ids") or [])],
            type=str(d.get("type", "other")),
            description=str(d.get("description", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class FinalValidation:
    document_id: str
    status: str = "pass"
    issues: List[FinalValidationIssue] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FinalValidation":
        return FinalValidation(
            document_id=str(d.get("document_id", "")),
            status=str(d.get("status", "pass")),
            issues=[FinalValidationIssue.from_dict(i) for i in (d.get("issues") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
        }

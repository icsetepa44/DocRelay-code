from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .common import to_dict


@dataclass
class SkillRecord:
    skill_id: str
    version: int
    status: str
    description: str
    applicability: List[str] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SkillRecord":
        return SkillRecord(
            skill_id=str(d["skill_id"]),
            version=int(d["version"]),
            status=str(d.get("status", "main")),
            description=str(d.get("description", "")),
            applicability=[str(x) for x in (d.get("applicability") or [])],
            instructions=[str(x) for x in (d.get("instructions") or [])],
            conflicts_with=[str(x) for x in (d.get("conflicts_with") or [])],
            provenance=dict(d.get("provenance") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    def extractor_view(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "instructions": list(self.instructions),
        }

    def retrieval_text(self) -> str:
        parts = [self.description, *self.applicability]
        reason = (self.provenance or {}).get("reason")
        if reason and self.status == "fallback":
            parts.append(str(reason))
        return "\n".join(p for p in parts if p)


@dataclass
class SelectedSkill:
    skill_id: str
    version: int
    relevance: float = 0.0
    reason: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SelectedSkill":
        return SelectedSkill(
            skill_id=str(d["skill_id"]),
            version=int(d.get("version", 1)),
            relevance=float(d.get("relevance", 0.0) or 0.0),
            reason=str(d.get("reason", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class SkillSelection:
    page_number: int
    skills: List[SelectedSkill] = field(default_factory=list)
    coverage: str = "sufficient"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SkillSelection":
        return SkillSelection(
            page_number=int(d.get("page_number", 0)),
            skills=[SelectedSkill.from_dict(s) for s in (d.get("skills") or [])],
            coverage=str(d.get("coverage", "sufficient")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "skills": [s.to_dict() for s in self.skills],
            "coverage": self.coverage,
        }

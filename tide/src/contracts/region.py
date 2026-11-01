from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .skill import SelectedSkill


@dataclass
class Region:
    region_id: str
    description: str
    source_reference: Dict[str, str] = field(default_factory=dict)
    continuation_context: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Region":
        cc = d.get("continuation_context")
        src = d.get("source_reference") or {}
        return Region(
            region_id=str(d["region_id"]),
            description=str(d.get("description", "")),
            source_reference={
                "markdown": str(src.get("markdown", "") or ""),
                "html": str(src.get("html", "") or ""),
            },
            continuation_context=(
                {"stitch_id": str(cc["stitch_id"])}
                if isinstance(cc, dict) and cc.get("stitch_id") else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "description": self.description,
            "source_reference": dict(self.source_reference),
            "continuation_context": self.continuation_context,
        }


@dataclass
class RegionAnalysis:
    page_number: int
    regions: List[Region] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RegionAnalysis":
        return RegionAnalysis(
            page_number=int(d.get("page_number", 0)),
            regions=[Region.from_dict(r) for r in (d.get("regions") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"page_number": self.page_number, "regions": [r.to_dict() for r in self.regions]}


@dataclass
class RegionSelection:
    region_id: str
    skills: List[SelectedSkill] = field(default_factory=list)
    coverage: str = "sufficient"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RegionSelection":
        return RegionSelection(
            region_id=str(d["region_id"]),
            skills=[SelectedSkill.from_dict(s) for s in (d.get("skills") or [])],
            coverage=str(d.get("coverage", "sufficient")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "skills": [s.to_dict() for s in self.skills],
            "coverage": self.coverage,
        }


@dataclass
class RegionSkillSelection:
    page_number: int
    regions: List[RegionSelection] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RegionSkillSelection":
        return RegionSkillSelection(
            page_number=int(d.get("page_number", 0)),
            regions=[RegionSelection.from_dict(r) for r in (d.get("regions") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"page_number": self.page_number, "regions": [r.to_dict() for r in self.regions]}

    def all_selected_skills(self) -> List[SelectedSkill]:
        seen = set()
        out: List[SelectedSkill] = []
        for reg in self.regions:
            for s in reg.skills:
                if s.skill_id in seen:
                    continue
                seen.add(s.skill_id)
                out.append(s)
        return out

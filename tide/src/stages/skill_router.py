"""
Skill Router (spec §8.3, §9).

- `route` (page mode): one page-wide SkillSelection from page-level candidates.
- `route_regions` (region mode): a per-region RegionSkillSelection; the router still sees
  the full page but decides skills + coverage independently for each region, choosing only
  from that region's RAG candidates. Per-region FALLBACK retrieval is a future hook (no
  fallback skills exist yet), so a region with coverage != sufficient simply proceeds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData
from src.contracts.region import RegionAnalysis, RegionSkillSelection
from src.contracts.skill import SkillRecord, SkillSelection
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import format_guidance, page_text_payload, payload_json, skill_candidate_view


def route(
    page: PageData,
    carry_state: CarryState,
    candidates: List[SkillRecord],
    image: Optional[Any] = None,
) -> SkillSelection:
    system = render_prompt(
        "skill_router",
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )
    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "carry_state": carry_state.to_dict(),
        "candidate_skills": [skill_candidate_view(r) for r in candidates],
    }
    user = (
        "Inspect the current page (image + page text) and "
        "choose the smallest set of the candidate skills that covers its structures.\n\n"
        + payload_json(payload)
    )
    obj = request_json(
        role="skill_router",
        system_prompt=system,
        user_prompt=user,
        schema_name="skill_selection",
        images=[image] if image is not None else None,
    )
    obj.setdefault("page_number", page.page_number)
    return SkillSelection.from_dict(obj)


def route_regions(
    page: PageData,
    carry_state: CarryState,
    region_analysis: RegionAnalysis,
    per_region_candidates: Dict[str, List[SkillRecord]],
    image: Optional[Any] = None,
) -> RegionSkillSelection:
    system = render_prompt(
        "skill_router_region",
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )
    regions_payload = []
    for region in region_analysis.regions:
        cands = per_region_candidates.get(region.region_id, [])
        regions_payload.append({
            "region_id": region.region_id,
            "description": region.description,
            "source_reference": region.source_reference,
            "continuation_context": region.continuation_context,
            "candidate_skills": [skill_candidate_view(r) for r in cands],
        })
    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "carry_state": carry_state.to_dict(),
        "regions": regions_payload,
    }
    user = (
        "For EACH region, keep only the candidate skills actually appropriate for it and "
        "judge that region's coverage. You still see the whole page.\n\n"
        + payload_json(payload)
    )
    obj = request_json(
        role="skill_router",
        system_prompt=system,
        user_prompt=user,
        schema_name="region_skill_selection",
        images=[image] if image is not None else None,
    )
    obj.setdefault("page_number", page.page_number)
    return RegionSkillSelection.from_dict(obj)

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData
from src.contracts.region import RegionAnalysis
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import format_guidance, page_text_payload, payload_json


def _normalize(raw: Dict[str, Any], page: PageData, carry_state: CarryState) -> RegionAnalysis:
    open_stitch = {u.stitch_id for u in carry_state.open_units}
    regions: List[Dict[str, Any]] = []
    for i, r in enumerate(raw.get("regions") or []):
        cc = r.get("continuation_context")
        stitch = cc.get("stitch_id") if isinstance(cc, dict) else None
        regions.append({
            "region_id": f"r{i + 1:03d}",
            "description": str(r.get("description", "")),
            "source_reference": r.get("source_reference") or {},
            "continuation_context": ({"stitch_id": stitch} if stitch in open_stitch else None),
        })
    return RegionAnalysis.from_dict({"page_number": page.page_number, "regions": regions})


def analyze(
    page: PageData,
    carry_state: CarryState,
    image: Optional[Any] = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> RegionAnalysis:
    system = render_prompt(
        "region_analyzer",
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )
    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "carry_state": carry_state.to_dict(),
    }
    user = (
        "Identify the logical/structural regions of this page for skill retrieval "
        "(do not extract).\n\n" + payload_json(payload)
    )
    raw = request_json(
        role="region_analyzer",
        system_prompt=system,
        user_prompt=user,
        schema_name="region_analysis",
        images=[image] if image is not None else None,
        model_override=model_override,
    )
    return _normalize(raw, page, carry_state)

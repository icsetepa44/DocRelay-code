from __future__ import annotations

from typing import Any, Dict, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData, PageExtraction
from src.contracts.super_validation import SuperValidationResult
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import format_guidance, page_text_payload, payload_json

CORRECTION_TRIGGERS = frozenset({
    "missing_text", "wrong_hierarchy", "split_clause", "merged_clause",
    "wrong_block_type", "wrong_reading_order",
})

NEXT_PAGE_SNIPPET_CHARS = 1200


def next_page_snippet(next_page: Optional[PageData]) -> Optional[str]:
    return next_page.markdown[:NEXT_PAGE_SNIPPET_CHARS] if next_page and next_page.markdown else None


def coverage_hint(underproduction_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not underproduction_record or "coverage" not in underproduction_record:
        return None
    return {
        "layout_regions": underproduction_record.get("n_regions"),
        "draft_blocks": underproduction_record.get("n_blocks"),
        "text_coverage": underproduction_record.get("coverage"),
        "under_produced": bool(underproduction_record.get("under_produced")),
    }


def validate(
    page: PageData,
    next_page: Optional[PageData],
    extraction: PageExtraction,
    carry_state: CarryState,
    image: Optional[Any] = None,
    model_override: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    draft_coverage: Optional[Dict[str, Any]] = None,
) -> SuperValidationResult:
    system = render_prompt(
        "super_validator",
        domain=domain,
        allowed_block_types=", ".join(config.ALLOWED_BLOCK_TYPES),
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )

    payload: Dict[str, Any] = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "next_page_snippet": next_page_snippet(next_page),
        "draft_extraction": extraction.to_dict(),
        "carry_state": carry_state.to_dict(),
    }
    if draft_coverage:
        payload["draft_coverage"] = draft_coverage

    user = (
        "Critique the following draft extraction for errors, missing content, and continuation.\n\n"
        + payload_json(payload)
    )

    raw = request_json(
        role="super_validator",
        system_prompt=system,
        user_prompt=user,
        schema_name="super_validation",
        images=[image] if image is not None else None,
        model_override=model_override,
    )

    return SuperValidationResult.from_dict(raw)


def needs_correction(super_validation: Optional[SuperValidationResult], extraction: PageExtraction) -> bool:
    if super_validation is None:
        return False
    if any(i.issue_type in CORRECTION_TRIGGERS for i in super_validation.issues):
        return True
    return not extraction.blocks

"""
FinalDecider stage (page corrector) — repairs the draft PageExtraction from the SuperValidator's
critique and the source page, and returns the complete corrected page.

Same stage as in the TIDE pipeline, except that the legal sub-point enforcer (`_enforce_subpoint_hierarchy`,
Bangla clause regexes) is deliberately left out: MPDocBench pages are a flat sequence of elements and
only headings parent headings (see prompts/extractor.md).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData, PageExtraction
from src.contracts.super_validation import SuperValidationResult
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import format_guidance, page_text_payload, payload_json
from src.stages.extractor import normalize_extraction
from src.stages.super_validator import next_page_snippet


def decide(
    page: PageData,
    next_page: Optional[PageData],
    extraction: PageExtraction,
    super_validation: SuperValidationResult,
    carry_state: CarryState,
    image: Optional[Any] = None,
    model_override: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
) -> PageExtraction:
    system = render_prompt(
        "final_decider",
        domain=domain,
        allowed_block_types=", ".join(config.ALLOWED_BLOCK_TYPES),
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )

    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "next_page_snippet": next_page_snippet(next_page),
        "draft_extraction": extraction.to_dict(),
        "super_validator_critique": super_validation.to_dict(),
        "carry_state": carry_state.to_dict(),
    }

    user = (
        "Repair and finalize the PageExtraction based on the SuperValidator critique and source page.\n\n"
        + payload_json(payload)
    )

    raw = request_json(
        role="final_decider",
        system_prompt=system,
        user_prompt=user,
        schema_name="page_extraction",
        images=[image] if image is not None else None,
        model_override=model_override,
    )

    return normalize_extraction(raw, page, carry_state)

from __future__ import annotations

from typing import Any, Dict, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData, PageExtraction
from src.contracts.super_validation import SuperValidationResult
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import format_guidance, page_text_payload, payload_json


def validate(
    page: PageData,
    next_page: Optional[PageData],
    extraction: PageExtraction,
    carry_state: CarryState,
    image: Optional[Any] = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> SuperValidationResult:
    system = render_prompt(
        "super_validator",
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )

    next_snippet = (next_page.markdown[:1200] if next_page and next_page.markdown else None)

    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "next_page_snippet": next_snippet,
        "draft_extraction": extraction.to_dict(),
        "carry_state": carry_state.to_dict(),
    }

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

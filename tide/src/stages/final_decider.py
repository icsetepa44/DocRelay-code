from __future__ import annotations

from typing import Any, Dict, Optional

from src.contracts.continuation import CarryState
from src.contracts.page import PageData, PageExtraction
from src.contracts.super_validation import SuperValidationResult
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import block_text, format_guidance, page_text_payload, payload_json
from src.stages.carry_rejection import MAINRULE, SUBPOINT
from src.stages.extractor import normalize_extraction


def _enforce_subpoint_hierarchy(extraction: PageExtraction, carry_state: CarryState) -> PageExtraction:
    last_main_rule_id = None
    open_main = next(
        (u for u in carry_state.open_units if getattr(u, "is_main_point", False) or u.block_type == "text_block"),
        None,
    )

    for b in extraction.blocks:
        txt = block_text(b).strip()
        if b.block_type == "text_block":
            if MAINRULE.match(txt):
                last_main_rule_id = b.block_id
            elif SUBPOINT.match(txt) and b.parent_block_id is None and b.stitch_id is None:
                if last_main_rule_id:
                    b.parent_block_id = last_main_rule_id
                elif open_main and open_main.stitch_id:
                    b.stitch_id = open_main.stitch_id

    return extraction


def decide(
    page: PageData,
    next_page: Optional[PageData],
    extraction: PageExtraction,
    super_validation: SuperValidationResult,
    carry_state: CarryState,
    image: Optional[Any] = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> PageExtraction:
    system = render_prompt(
        "final_decider",
        ocr_format_note=format_guidance(config.OCR_INPUT_FORMAT),
    )

    next_snippet = (next_page.markdown[:1200] if next_page and next_page.markdown else None)

    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "next_page_snippet": next_snippet,
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

    normalized = normalize_extraction(raw, page, carry_state)
    return _enforce_subpoint_hierarchy(normalized, carry_state)

from __future__ import annotations

from typing import Any, List, Optional

from src.contracts.continuation import CarryState, ContinuationDecision
from src.contracts.page import PageData, PageExtraction
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import page_text_payload, payload_json


def judge(
    page: PageData,
    next_page: PageData,
    extraction: PageExtraction,
    carry_state: CarryState,
    uncertain_block_ids: List[str],
    image: Optional[Any] = None,
    next_image: Optional[Any] = None,
) -> ContinuationDecision:
    system = render_prompt("continuation_judge")
    fmt = config.OCR_INPUT_FORMAT
    payload = {
        "page_number": page.page_number,
        **{f"current_{k}": v for k, v in page_text_payload(page, fmt).items()},
        **{f"next_{k}": v for k, v in page_text_payload(next_page, fmt).items()},
        "carry_state": carry_state.to_dict(),
        "extraction": extraction.to_dict(),
        "block_ids_to_judge": list(uncertain_block_ids),
    }
    user = (
        "Judge continuation for ONLY the listed block_ids, comparing the end of the "
        "current page with the start of the next page.\n\n" + payload_json(payload)
    )
    images = [im for im in (image, next_image) if im is not None] or None
    obj = request_json(
        role="continuation_judge",
        system_prompt=system,
        user_prompt=user,
        schema_name="continuation_decision",
        images=images,
    )
    obj.setdefault("page_number", page.page_number)
    return ContinuationDecision.from_dict(obj)

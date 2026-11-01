"""
Validator (spec §8.5, §11). Independently checks whether a PageExtraction faithfully
represents the source page. In v1 this is REPORT-ONLY, so it is parsed LENIENTLY:
unknown issue types are coerced to "other" rather than failing, and a missing/odd
status is inferred from whether issues were reported. The orchestrator persists and
logs the result but never branches into repair (deferred Phase B).
"""

from __future__ import annotations

from typing import Any, Optional

from src.contracts.continuation import CarryState
from src.contracts.common import PAGE_ISSUE_TYPES, VALIDATION_STATUSES
from src.contracts.page import PageData, PageExtraction
from src.contracts.validation import PageValidation, ValidationIssue
from src.core import config
from src.llm.client import chat_completion
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import build_request_config, extract_json_object
from src.stages._common import page_text_payload, payload_json


def _coerce(obj: dict, page_number: int) -> PageValidation:
    issues = []
    for it in obj.get("issues") or []:
        itype = str(it.get("type", "other"))
        if itype not in PAGE_ISSUE_TYPES:
            itype = "other"
        issues.append(ValidationIssue(
            block_ids=[str(b) for b in (it.get("block_ids") or [])],
            type=itype,
            description=str(it.get("description", "")),
        ))
    status = str(obj.get("status", "")).strip().lower()
    if status not in VALIDATION_STATUSES:
        status = "fail" if issues else "pass"
    return PageValidation(page_number=int(obj.get("page_number", page_number)),
                          status=status, issues=issues)


def validate(
    page: PageData,
    extraction: PageExtraction,
    carry_state: CarryState,
    image: Optional[Any] = None,
) -> PageValidation:
    system = render_prompt("validator")
    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, config.OCR_INPUT_FORMAT),
        "carry_state": carry_state.to_dict(),
        "extraction": extraction.to_dict(),
    }
    user = (
        "Validate this PageExtraction against the source page. Report problems only.\n\n"
        + payload_json(payload)
    )
    raw = chat_completion(
        system_prompt=system,
        user_prompt=user,
        config=build_request_config("validator"),
        images=[image] if image is not None else None,
    )
    return _coerce(extract_json_object(raw), page.page_number)

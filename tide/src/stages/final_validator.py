"""
Final Validator (spec §8.15, §18). Checks the assembled FinalDocument for cross-page
reconstruction errors (duplicated boundary text, missing continuation, wrong stitch
grouping, unresolved open units, …). Report-only in v1.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.contracts.final_document import FinalDocument
from src.contracts.validation import FinalValidation, FinalValidationIssue
from src.core import config
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import payload_json


def _coverage_gate(page_summaries: List[Dict[str, Any]]):
    """Deterministic empty-page check (C2). Returns (empty, total, ratio, tripped)."""
    total = len(page_summaries)
    empty = sum(1 for s in page_summaries if s.get("empty_nontrivial"))
    ratio = (empty / total) if total else 0.0
    return empty, total, ratio, (total > 0 and ratio > config.MAX_EMPTY_PAGE_RATIO)


def validate(
    document_id: str,
    final_document: FinalDocument,
    page_summaries: List[Dict[str, Any]],
) -> FinalValidation:
    system = render_prompt("final_validator")
    payload = {
        "document_id": document_id,
        "final_document": final_document.to_dict(),
        "page_summaries": page_summaries,
    }
    user = (
        "Validate the assembled document for cross-page reconstruction errors. "
        "Report problems only.\n\n" + payload_json(payload)
    )
    obj = request_json(
        role="final_validator",
        system_prompt=system,
        user_prompt=user,
        schema_name="final_validation",
    )
    obj.setdefault("document_id", document_id)
    result = FinalValidation.from_dict(obj)

    # C2: deterministic coverage gate. If too many NON-TRIVIAL pages came back empty
    # (dropped/failed pages), FLAG the document as failed — never silently "pass".
    empty, total, ratio, tripped = _coverage_gate(page_summaries)
    if tripped:
        result.status = "fail"
        result.issues.insert(0, FinalValidationIssue(
            unit_ids=[],
            type="low_coverage",
            description=(f"{empty}/{total} non-trivial pages produced zero blocks "
                         f"(empty ratio {ratio:.2f} > {config.MAX_EMPTY_PAGE_RATIO}); "
                         "the document is likely missing dropped/failed pages."),
        ))
    return result

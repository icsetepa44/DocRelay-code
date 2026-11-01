from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.contracts.block import Block
from src.contracts.common import SkillReference
from src.contracts.continuation import BlockContinuation, CarryState, ContinuationDecision
from src.contracts.page import PageData, PageExtraction
from src.contracts.skill import SkillRecord
from src.core import config
from src.core import ids
from src.llm.prompt_loader import render_prompt
from src.llm.structured_output import request_json
from src.stages._common import format_guidance, page_text_payload, payload_json


def _skill_instruction_view(record: SkillRecord) -> Dict[str, Any]:
    return record.extractor_view()


def normalize_extraction(
    raw: Dict[str, Any],
    page: PageData,
    carry_state: CarryState,
) -> PageExtraction:
    doc = page.document_id
    page_no = page.page_number
    raw_blocks = list(raw.get("blocks") or [])
    incoming_open = {u.stitch_id for u in carry_state.open_units}

    local_to_canonical: Dict[str, str] = {}
    for i, rb in enumerate(raw_blocks):
        local_id = str(rb.get("block_id") or f"b{i + 1}")
        local_to_canonical.setdefault(local_id, ids.block_id(doc, page_no, i + 1))

    blocks: List[Block] = []
    for i, rb in enumerate(raw_blocks):
        canonical = ids.block_id(doc, page_no, i + 1)

        btype = str(rb.get("block_type") or "text_block")
        if btype not in config.ALLOWED_BLOCK_TYPES:
            btype = "text_block"

        parent_local = rb.get("parent_block_id")
        parent = local_to_canonical.get(str(parent_local)) if parent_local else None
        if parent == canonical:
            parent = None

        model_stitch = rb.get("stitch_id")
        stitch = model_stitch if (model_stitch and model_stitch in incoming_open) else None

        blocks.append(
            Block(
                block_id=canonical,
                block_type=btype,
                parent_block_id=parent,
                stitch_id=stitch,
                content=dict(rb.get("content") or {}),
                skills_used=SkillReference.list_from(rb.get("skills_used")),
            )
        )

    canonical_by_id = {b.block_id: b for b in blocks}

    decisions: List[BlockContinuation] = []
    for d in (raw.get("continuation") or {}).get("decisions") or []:
        bl = local_to_canonical.get(str(d.get("block_id")))
        if not bl or bl not in canonical_by_id:
            continue
        decisions.append(
            BlockContinuation(
                block_id=bl,
                stitch_id=canonical_by_id[bl].stitch_id,
                decision=str(d.get("decision", "uncertain")),
                evidence=[str(e) for e in (d.get("evidence") or [])],
                is_main_point=(bool(d["is_main_point"]) if d.get("is_main_point") is not None else None),
                expected_next_main_point=(
                    str(d["expected_next_main_point"]) if d.get("expected_next_main_point") else None),
                continuation_hint=(str(d["continuation_hint"]) if d.get("continuation_hint") else None),
            )
        )

    from src.stages.carry_rejection import MAINRULE, SUBPOINT
    from src.stages._common import block_text

    last_main_rule_id = None
    open_main = next(
        (u for u in carry_state.open_units if getattr(u, "is_main_point", False) or u.block_type == "text_block"),
        None,
    )
    for b in blocks:
        txt = block_text(b).strip()
        if b.block_type == "text_block":
            if MAINRULE.match(txt):
                last_main_rule_id = b.block_id
            elif SUBPOINT.match(txt) and b.parent_block_id is None and b.stitch_id is None:
                if last_main_rule_id:
                    b.parent_block_id = last_main_rule_id
                elif open_main and open_main.stitch_id:
                    b.stitch_id = open_main.stitch_id

    return PageExtraction(
        page_number=page_no,
        blocks=blocks,
        continuation=ContinuationDecision(page_number=page_no, decisions=decisions),
    )


def extract(
    page: PageData,
    next_page: Optional[PageData],
    carry_state: CarryState,
    skills: List[SkillRecord],
    image: Optional[Any] = None,
    model_override: Optional[Dict[str, Any]] = None,
    regions: Optional[List[Dict[str, Any]]] = None,
    ocr_format: Optional[str] = None,
) -> PageExtraction:
    fmt = ocr_format or config.OCR_INPUT_FORMAT
    system = render_prompt(
        "extractor",
        allowed_block_types=", ".join(config.ALLOWED_BLOCK_TYPES),
        ocr_format_note=format_guidance(fmt),
    )
    payload = {
        "page_number": page.page_number,
        **page_text_payload(page, fmt),
        "next_page": (
            page_text_payload(next_page, fmt) if next_page is not None else None
        ),
        "carry_state": carry_state.to_dict(),
        "selected_skills": [_skill_instruction_view(r) for r in skills],
    }
    if regions:
        payload["regions"] = regions
    user = (
        "Extract ONLY the current page into PageExtraction JSON. Use the next_page text "
        "only to decide continuation. Follow the selected skill instructions.\n\n"
        + payload_json(payload)
    )
    raw = request_json(
        role="extractor",
        system_prompt=system,
        user_prompt=user,
        schema_name="page_extraction",
        images=[image] if image is not None else None,
        model_override=model_override,
    )
    return normalize_extraction(raw, page, carry_state)

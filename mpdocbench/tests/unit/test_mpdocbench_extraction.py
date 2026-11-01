import dataclasses

import pytest

from src.contracts.continuation import BlockContinuation, CarryState, OpenUnit
from src.contracts.page import PageData, PageExtraction
from src.core.ids import StitchIdAllocator
from src.llm.prompt_loader import render_prompt
from src.stages import assembler, carry_builder, extractor
from src.stages._common import block_text, format_guidance, layout_guidance


def test_b1_fields_removed_from_contracts():
    bc_fields = {f.name for f in dataclasses.fields(BlockContinuation)}
    ou_fields = {f.name for f in dataclasses.fields(OpenUnit)}
    for f in ("is_main_point", "expected_next_main_point", "continuation_hint"):
        assert f not in bc_fields and f not in ou_fields
    bc = BlockContinuation.from_dict({"block_id": "b", "decision": "continues",
                                      "is_main_point": True, "expected_next_main_point": "3.",
                                      "continuation_hint": "x"})
    assert "is_main_point" not in bc.to_dict()
    ou = OpenUnit.from_dict({"stitch_id": "s1", "last_block_id": "b", "block_type": "text_block",
                             "state": "", "is_main_point": True})
    assert "is_main_point" not in ou.to_dict()
    with pytest.raises(TypeError):
        OpenUnit(stitch_id="s1", last_block_id="b", block_type="text_block", state="",
                 is_main_point=True)


def _raw():
    return {"page_number": 1, "blocks": [
        {"block_id": "h1", "block_type": "heading", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "Intro", "level": 2}, "skills_used": []},
        {"block_id": "p1", "block_type": "text_block", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "A paragraph."}, "skills_used": []},
        {"block_id": "fig", "block_type": "figure", "parent_block_id": None, "stitch_id": None,
         "content": {"description": "a bar chart"}, "skills_used": []},
        {"block_id": "eq", "block_type": "equation", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "$$E=mc^2$$"}, "skills_used": []},
        {"block_id": "cap", "block_type": "caption", "parent_block_id": "fig", "stitch_id": None,
         "content": {"text": "Figure 1."}, "skills_used": []},
        {"block_id": "lst", "block_type": "list", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "a\nb\nc"}, "skills_used": []},
        {"block_id": "cd", "block_type": "code", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "print(1)"}, "skills_used": []},
        {"block_id": "old", "block_type": "image_or_seal", "parent_block_id": None,
         "stitch_id": None, "content": {"description": "seal"}, "skills_used": []},
    ], "continuation": {"page_number": 1, "decisions": []}}


def test_normalizer_keeps_general_types_and_level_and_coerces_removed():
    page = PageData("doc", 1, "x.png", "md", "h")
    pe = extractor.normalize_extraction(_raw(), page, CarryState.empty())
    assert [b.block_type for b in pe.blocks] == [
        "heading", "text_block", "figure", "equation", "caption", "list", "code",
        "text_block",
    ]
    assert pe.blocks[0].content.get("level") == 2
    assert pe.blocks[4].parent_block_id == pe.blocks[2].block_id
    assert block_text(pe.blocks[0]) == "Intro"
    assert block_text(pe.blocks[2]) == "a bar chart"
    assert block_text(pe.blocks[3]) == "$$E=mc^2$$"


def test_simple_cross_page_single_element_stitch():
    p0 = PageExtraction.from_dict({"page_number": 0, "blocks": [
        {"block_id": "doc:p0000:b0001", "block_type": "text_block", "parent_block_id": None,
         "stitch_id": None, "content": {"text": "a paragraph that is cut"}, "skills_used": []}],
        "continuation": {"page_number": 0, "decisions": [
            {"block_id": "doc:p0000:b0001", "decision": "continues", "evidence": ["cut"]}]}})
    cs = carry_builder.build_carry_state(p0, p0.continuation, StitchIdAllocator("doc"))
    sid = cs.open_units[0].stitch_id
    assert p0.blocks[0].stitch_id == sid
    p1 = PageExtraction.from_dict({"page_number": 1, "blocks": [
        {"block_id": "doc:p0001:b0001", "block_type": "text_block", "parent_block_id": None,
         "stitch_id": sid, "content": {"text": "and finishes here."}, "skills_used": []}]})
    fd = assembler.assemble("doc", [p0, p1])
    assert [u.unit_id for u in fd.units] == [sid]
    merged = fd.units[0]
    assert isinstance(merged.content, list) and len(merged.content) == 2
    assert merged.source_block_ids == ["doc:p0000:b0001", "doc:p0001:b0001"]


def test_retry_note_injected_into_user_prompt(monkeypatch):
    captured = {}

    def fake_request_json(**kwargs):
        captured["user"] = kwargs.get("user_prompt", "")
        return {"page_number": 1, "blocks": [], "continuation": {"page_number": 1, "decisions": []}}

    monkeypatch.setattr(extractor, "request_json", fake_request_json)
    page = PageData("doc", 1, "x.png", "some markdown text", "<p>h</p>")
    note = "you returned only 1 block but the page has ~18 regions — extract every element"

    extractor.extract(page, None, CarryState.empty(), [], retry_note=note)
    assert note in captured["user"]

    extractor.extract(page, None, CarryState.empty(), [])
    assert "extract every element" not in captured["user"]


def test_prompts_render_general_and_b1_free():
    ext = render_prompt("extractor", allowed_block_types="text_block, heading, table",
                        ocr_format_note=format_guidance("mixed"))
    assert "{{" not in ext
    for legacy in ("is_main_point", "expected_next_main_point", "continuation_hint",
                   "image_or_seal", "তফসিল"):
        assert legacy not in ext
    assert "level" in ext
    judge = render_prompt("continuation_judge")
    assert "{{" not in judge and "expected_next_main_point" not in judge
    assert "{{" not in render_prompt("region_analyzer", ocr_format_note=format_guidance("mixed"),
                                     layout_note=layout_guidance(True))

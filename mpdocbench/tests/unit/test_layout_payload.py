"""layout payload (LAPDoc SpatialFormat) helpers + region_analyzer injection. No LLM.

Covers ordering / typing / normalization of `layout_regions`, the whitespace SpatialFormat rendering
of `layout_payload`, the `{{layout_note}}` prompt token, and that `region_analyzer.analyze` injects the
layout string into what the model sees (and omits it for a layout-less page)."""

import pytest

from src.contracts.continuation import CarryState
from src.contracts.page import PageData
from src.llm import structured_output as so
from src.llm.prompt_loader import render_prompt
from src.stages import _common, region_analyzer


def _layout():
    return {"page_num": 1, "page_box": [0, 0, 1000, 1000], "chunks": [
        {"bbox": [50, 100, 150, 130], "label": "Section-Header", "content": "<h1>Title</h1>"},
        {"bbox": [50, 200, 500, 260], "label": "Text", "content": "<p>Body   text  here.</p>"},
        {"bbox": [520, 200, 950, 400], "label": "Image", "content": "a chart"},
        {"bbox": [50, 900, 300, 940], "label": "Weird-Label", "content": "footer-ish"},
    ]}


# --------------------------------------------------------------------------- layout_regions
def test_layout_regions_ordering_typing_normalization():
    regs = _common.layout_regions(_layout())
    assert [r["region_id"] for r in regs] == ["L001", "L002", "L003", "L004"]  # reading order kept
    assert [r["type"] for r in regs] == ["heading", "text", "figure", "other"]  # label map + fallback
    # page_box 1000x1000 -> box == bbox; HTML stripped + whitespace collapsed
    assert regs[0]["box"] == [50, 100, 150, 130] and regs[0]["text"] == "Title"
    assert regs[1]["box"] == [50, 200, 500, 260] and regs[1]["text"] == "Body text here."
    assert regs[2]["text"] == "a chart"


def test_layout_regions_scales_non_1000_pagebox():
    layout = {"page_box": [0, 0, 500, 2000],
              "chunks": [{"bbox": [50, 100, 150, 300], "label": "Text", "content": "x"}]}
    assert _common.layout_regions(layout)[0]["box"] == [100, 50, 300, 150]


def test_layout_regions_clamps_and_snippet_cap():
    layout = {"page_box": [0, 0, 100, 100],
              "chunks": [{"bbox": [-10, 0, 200, 50], "label": "Text", "content": "z" * 500}]}
    r = _common.layout_regions(layout)[0]
    assert r["box"] == [0, 0, 1000, 500]                 # clamped to [0,1000]
    assert len(r["text"]) == _common._LAYOUT_SNIPPET_CHARS  # snippet capped


def test_layout_regions_edge_cases():
    assert _common.layout_regions({}) == []
    assert _common.layout_regions({"chunks": [{"bbox": [1, 2, 3, 4], "label": "Text"}]}) == []  # no page_box
    assert _common.layout_regions({"page_box": [0, 0, 0, 100], "chunks": [
        {"bbox": [1, 2, 3, 4], "label": "Text"}]}) == []                                        # zero width
    # a chunk with <4 bbox coords is skipped, others survive
    regs = _common.layout_regions({"page_box": [0, 0, 100, 100], "chunks": [
        {"bbox": [1, 2], "label": "Text", "content": "bad"},
        {"bbox": [1, 2, 3, 4], "label": "Text", "content": "ok"}]})
    assert len(regs) == 1 and regs[0]["region_id"] == "L001" and regs[0]["text"] == "ok"


# --------------------------------------------------------------------------- layout_payload
def test_layout_payload_spatial_rendering():
    s = _common.layout_payload(_layout())
    assert isinstance(s, str) and s.splitlines()[0].startswith("Page layout")
    assert "[heading 50,100-150,130] Title" in s
    # indentation grows with x0: the figure (x0=520) is indented far more than the heading (x0=50)
    fig_line = next(ln for ln in s.splitlines() if "[figure" in ln)
    head_line = next(ln for ln in s.splitlines() if "[heading" in ln)
    assert (len(fig_line) - len(fig_line.lstrip())) > (len(head_line) - len(head_line.lstrip()))
    assert (len(head_line) - len(head_line.lstrip())) == 50 // _common._LAYOUT_INDENT_DIV
    # a big vertical gap (y1=400 -> next y0=900) inserts a blank line
    assert "\n\n" in s


def test_layout_payload_empty():
    assert _common.layout_payload({}) == ""
    assert _common.layout_payload({"page_box": [0, 0, 0, 0], "chunks": []}) == ""


# --------------------------------------------------------------------------- layout_guidance
def test_layout_guidance_present_vs_absent():
    present, absent = _common.layout_guidance(True), _common.layout_guidance(False)
    assert present != absent and "layout" in present.lower() and absent


# --------------------------------------------------------------------------- prompt token
def test_region_prompt_substitutes_layout_note():
    rendered = render_prompt("region_analyzer",
                             ocr_format_note=_common.format_guidance("mixed"),
                             layout_note=_common.layout_guidance(True))
    assert "{{" not in rendered                       # both tokens substituted
    assert "provides no bounding boxes" not in rendered  # the old contradicting line is gone
    assert "reading order" in rendered.lower()


# --------------------------------------------------------------------------- analyze injection
def test_analyze_injects_layout(monkeypatch):
    cap = {}

    def fake_chat(**kw):
        cap.update(kw)
        return '{"page_number": 1, "regions": []}'

    monkeypatch.setattr(so, "chat_completion", fake_chat)
    monkeypatch.setattr(so, "build_request_config", lambda role, model_override=None: object())

    page = PageData("doc", 1, "x.png", "md", "html", layout=_layout())
    region_analyzer.analyze(page, CarryState.empty())
    assert '"layout"' in cap["user_prompt"] and "[heading 50,100-150,130]" in cap["user_prompt"]
    assert "reading order" in cap["system_prompt"].lower()  # layout_note landed in the system prompt

    # back-compat: a page with no layout -> no `layout` key, absent-note in system prompt
    cap.clear()
    region_analyzer.analyze(PageData("doc", 1, "x.png", "md", "html"), CarryState.empty())
    assert '"layout"' not in cap["user_prompt"]

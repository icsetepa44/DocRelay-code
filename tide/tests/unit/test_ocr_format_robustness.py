"""Mixed-OCR fragility fix (FAILURE_ANALYSIS_V8): per-format prompt notes, nullable
region source_reference, and null-string schema healing."""

import json

from src.llm import structured_output as so
from src.llm.prompt_loader import render_prompt
from src.stages import _common


def test_format_guidance_per_format():
    notes = {f: _common.format_guidance(f) for f in ("markdown", "html", "both", "mixed")}
    assert len(set(notes.values())) == 4                     # all four are distinct
    assert "HTML" in notes["html"]
    assert "Markdown" in notes["markdown"]
    assert "inline as HTML" in notes["mixed"]
    # unknown format falls back to the 'both' note (matches page_text_payload's guard)
    assert _common.format_guidance("bogus") == _common.format_guidance("both")


def test_region_analysis_schema_tolerates_null_and_missing_html():
    obj = {"page_number": 1, "regions": [
        {"region_id": "r001", "description": "d", "source_reference": {"markdown": "x", "html": None}}]}
    so.validate(obj, "region_analysis")                      # null html no longer crashes
    obj["regions"][0]["source_reference"] = {"markdown": "x"}        # missing html
    so.validate(obj, "region_analysis")
    obj["regions"][0]["source_reference"] = {"markdown": None, "html": None}  # both null
    so.validate(obj, "region_analysis")


def test_coerce_null_strings_heals_only_null_where_string():
    obj = {"page_number": 1, "regions": [{"region_id": "r1", "description": None}]}
    assert so._coerce_null_strings(obj, "region_analysis") is True
    assert obj["regions"][0]["description"] == ""
    so.validate(obj, "region_analysis")                      # valid after coercion
    # nothing to heal on an already-valid object
    assert so._coerce_null_strings({"page_number": 1, "regions": []}, "region_analysis") is False


def test_request_json_heals_null_string_without_reask(monkeypatch):
    bad = json.dumps({"page_number": 1, "regions": [{"region_id": "r1", "description": None}]})
    calls = {"n": 0}

    def fake_chat(**_kw):
        calls["n"] += 1
        return bad

    monkeypatch.setattr(so, "build_request_config", lambda role, model_override=None: object())
    monkeypatch.setattr(so, "chat_completion", fake_chat)
    out = so.request_json(role="region_analyzer", system_prompt="s", user_prompt="u",
                          schema_name="region_analysis")
    assert out["regions"][0]["description"] == ""            # healed in-place
    assert calls["n"] == 1                                   # no corrective re-ask needed


def test_prompts_substitute_ocr_format_note():
    note = _common.format_guidance("mixed")
    for name in ("region_analyzer", "extractor", "skill_router", "skill_router_region"):
        kwargs = {"ocr_format_note": note}
        if name == "extractor":
            kwargs["allowed_block_types"] = "text_block"
        rendered = render_prompt(name, **kwargs)
        assert "{{ocr_format_note}}" not in rendered         # token was substituted
        assert "inline as HTML" in rendered                  # the mixed note landed

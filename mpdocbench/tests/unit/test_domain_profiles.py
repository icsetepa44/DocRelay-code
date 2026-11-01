"""domain threading + per-domain profile machinery. Hermetic — no LLM, no scored run.

Covers domain resolution, the (empty-by-default) profile resolver, the `render_prompt` override-file
selection + `{{domain_profile}}` token, the retrieval weight/pin hooks, and that the extractor /
region_analyzer forward `domain` into `render_prompt`. Ships behavior-neutral: the 11 shipped profiles
are empty, so with the real config every resolver returns empties and prompts render the token to ''."""

import pytest

from src.contracts.continuation import CarryState
from src.contracts.page import PageData
from src.core import config
from src.llm import prompt_loader
from src.llm.prompt_loader import load_prompt, render_prompt
from src.registry import skill_retriever
from src.stages import extractor, region_analyzer


# --------------------------------------------------------------------------- domain resolution
def test_domain_for_uses_the_stem_prefix():
    # MPDocBench stems are "<domain>_<lang>_<hash>"
    assert config.domain_for("annualReport_eng_00000000") == "annualReport"
    assert config.domain_for("newspaper_chn_00000000") == "newspaper"
    assert config.domain_for("prospectus_eng_zzzzzzzz") == "prospectus"
    assert config.domain_for("researchReport_eng_x") == "researchReport"
    assert config.domain_for("") == ""


def test_domain_profile_empty_by_default():
    empty = {"guidance": "", "pin_skills": [], "weight_skills": {}}
    assert config.domain_profile(None) == empty
    assert config.domain_profile("nonexistent-domain") == empty
    # the domains left empty stay empty (unchanged behavior)
    for dom in ("annualReport", "book", "examPaper", "goverment", "paper",
                "prospectus", "researchReport", "textBook"):
        assert config.domain_profile(dom) == empty


def test_sc2_filled_domains_have_guidance_and_valid_pins():
    # newspaper/magazine/manualGuide carry a guidance paragraph + a couple of pins.
    from src.registry.skill_registry import get_registry
    ids = {r.skill_id for r in get_registry().by_status("main")}
    expect = {"newspaper": {"caption", "figure"}, "magazine": {"caption", "figure"},
              "manualGuide": {"list", "caption"}}
    for dom, pins in expect.items():
        p = config.domain_profile(dom)
        assert p["guidance"] and set(p["pin_skills"]) == pins
        assert set(p["pin_skills"]) <= ids                       # pins name real skills (typo guard)


def test_sc2_guidance_renders_into_prompts_only_for_filled_domains():
    # the guidance opens with a domain-unique phrase ("This is a <kind> page"); the base prompt's own
    # "own block" wording is NOT a valid marker (it's present regardless of domain).
    for name, kw in (("extractor", {"allowed_block_types": "x", "ocr_format_note": "y"}),
                     ("region_analyzer", {"ocr_format_note": "y", "layout_note": "z"})):
        assert "newspaper page" in render_prompt(name, domain="newspaper", **kw).lower()
    # an unfilled domain renders the token to '' (no guidance leaked, no leftover token)
    book_ext = render_prompt("extractor", domain="book", allowed_block_types="x", ocr_format_note="y")
    assert "newspaper page" not in book_ext.lower() and "{{" not in book_ext


def test_sc2_domain_pins_surface_in_retrieval(monkeypatch):
    _kw(monkeypatch)
    pins = config.domain_profile("newspaper")["pin_skills"]
    got = {r.skill_id for r in skill_retriever.retrieve_skills(
        "a paragraph of prose", ["main"], 1, pin_skill_ids=pins)}
    assert {"caption", "figure"} <= got                          # domain pins are force-included


def test_domain_profile_merges_override(monkeypatch):
    monkeypatch.setattr(config, "DOMAIN_PROFILES", {
        "defaults": {"guidance": "", "pin_skills": [], "weight_skills": {}},
        "domains": {"book": {"guidance": "Books: prefer prose.", "pin_skills": ["heading"],
                             "weight_skills": {"table": 2.0}}}})
    p = config.domain_profile("book")
    assert p == {"guidance": "Books: prefer prose.", "pin_skills": ["heading"],
                 "weight_skills": {"table": 2.0}}
    assert config.domain_profile("magazine")["guidance"] == ""  # unlisted -> defaults


# --------------------------------------------------------------------------- render_prompt token
def test_render_fills_domain_profile_token_empty_when_no_domain():
    # extractor.md / region_analyzer.md now carry {{domain_profile}}; with no domain it renders to ''
    ext = render_prompt("extractor", allowed_block_types="text_block", ocr_format_note="x")
    reg = render_prompt("region_analyzer", ocr_format_note="x", layout_note="y")
    assert "{{" not in ext and "{{" not in reg  # no leftover token (protects existing assertions)


def test_render_injects_domain_guidance(monkeypatch):
    monkeypatch.setattr(config, "domain_profile",
                        lambda d: {"guidance": "GUIDE_MARKER_XYZ", "pin_skills": [], "weight_skills": {}})
    ext = render_prompt("extractor", domain="book", allowed_block_types="x", ocr_format_note="y")
    assert "GUIDE_MARKER_XYZ" in ext


def test_render_prompt_domain_override_file(tmp_path, monkeypatch):
    (tmp_path / "foo.md").write_text("BASE {{domain_profile}}", encoding="utf-8")
    (tmp_path / "foo__book.md").write_text("OVERRIDE for book", encoding="utf-8")
    monkeypatch.setattr(config, "PROMPTS_DIR", tmp_path)
    monkeypatch.setattr(config, "domain_profile",
                        lambda d: {"guidance": "", "pin_skills": [], "weight_skills": {}})
    load_prompt.cache_clear()
    try:
        assert render_prompt("foo") == "BASE "                       # base, token -> ''
        assert render_prompt("foo", domain="book") == "OVERRIDE for book"  # override file used
        assert render_prompt("foo", domain="nodom") == "BASE "        # no override file -> base fallback
    finally:
        load_prompt.cache_clear()


# --------------------------------------------------------------------------- retrieval pin/weight
def _kw(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_SCORER", "keyword")  # stdlib scorer, no torch


def test_retrieve_weight_boost_changes_ranking(monkeypatch):
    _kw(monkeypatch)
    q = "a data table with rows and columns"
    base = [r.skill_id for r in skill_retriever.retrieve_skills(q, ["main"], 1)]
    assert base == ["table"]                                        # table wins on its own
    boosted = [r.skill_id for r in skill_retriever.retrieve_skills(
        q, ["main"], 1, weight_skill_ids={"figure": 1000.0})]
    assert boosted == ["figure"]                                    # a large boost overrides ranking


def test_retrieve_pin_and_empty_hooks_are_noops(monkeypatch):
    _kw(monkeypatch)
    q = "a data table with rows and columns"
    pinned = {r.skill_id for r in skill_retriever.retrieve_skills(
        q, ["main"], 1, pin_skill_ids=["figure"])}
    assert "table" in pinned and "figure" in pinned                 # top-1 + forced pin
    # empty pin/weight == no arguments (behavior-neutral default)
    a = [r.skill_id for r in skill_retriever.retrieve_skills(q, ["main"], 3)]
    b = [r.skill_id for r in skill_retriever.retrieve_skills(
        q, ["main"], 3, pin_skill_ids=[], weight_skill_ids={})]
    assert a == b


# --------------------------------------------------------------------------- stage threading
def test_extractor_forwards_domain_to_render(monkeypatch):
    cap = {}
    monkeypatch.setattr(extractor, "render_prompt",
                        lambda name, **kw: cap.update(name=name, **kw) or "SYS")
    monkeypatch.setattr(extractor, "request_json",
                        lambda **kw: {"page_number": 1, "blocks": [],
                                      "continuation": {"page_number": 1, "decisions": []}})
    extractor.extract(PageData("d", 1, "x.png", "md", "html"), None, CarryState.empty(), [],
                      domain="book")
    assert cap["name"] == "extractor" and cap["domain"] == "book"


def test_region_analyzer_forwards_domain_to_render(monkeypatch):
    cap = {}
    monkeypatch.setattr(region_analyzer, "render_prompt",
                        lambda name, **kw: cap.update(name=name, **kw) or "SYS")
    monkeypatch.setattr(region_analyzer, "request_json",
                        lambda **kw: {"page_number": 1, "regions": []})
    region_analyzer.analyze(PageData("d", 1, "x.png", "md", "html"), CarryState.empty(),
                            domain="book")
    assert cap["name"] == "region_analyzer" and cap["domain"] == "book"

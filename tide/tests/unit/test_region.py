import json

from jsonschema import Draft202012Validator

from src.contracts.continuation import CarryState
from src.contracts.page import PageData
from src.contracts.region import RegionAnalysis, RegionSkillSelection
from src.core import config
from src.stages import region_analyzer, skill_router


def _schema(name):
    return json.loads((config.SCHEMAS_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def test_region_analysis_round_trip_and_schema():
    d = {"page_number": 3, "regions": [
        {"region_id": "r001", "description": "x",
         "source_reference": {"markdown": "m", "html": "h"}, "continuation_context": None}]}
    ra = RegionAnalysis.from_dict(d)
    assert ra.to_dict() == d
    Draft202012Validator(_schema("region_analysis")).validate(ra.to_dict())


def test_region_skill_selection_union_and_schema():
    d = {"page_number": 3, "regions": [
        {"region_id": "r001", "skills": [{"skill_id": "numbered_point", "version": 1}],
         "coverage": "sufficient"},
        {"region_id": "r002", "skills": [{"skill_id": "numbered_point", "version": 1},
                                          {"skill_id": "table", "version": 1}],
         "coverage": "partial"}]}
    s = RegionSkillSelection.from_dict(d)
    Draft202012Validator(_schema("region_skill_selection")).validate(s.to_dict())
    assert [x.skill_id for x in s.all_selected_skills()] == ["numbered_point", "table"]  # de-duped


def test_region_skill_selection_schema_rejects_bad_coverage():
    bad = {"page_number": 1, "regions": [{"region_id": "r001", "skills": [], "coverage": "BAD"}]}
    assert list(Draft202012Validator(_schema("region_skill_selection")).iter_errors(bad))


def test_region_analyzer_normalize_reassigns_ids_and_validates_stitch():
    carry = CarryState.from_dict({"from_page": 2, "open_units": [
        {"stitch_id": "doc:s000007", "last_block_id": "b", "block_type": "text_block",
         "state": "s", "continuation_evidence": [], "skills": []}]})
    raw = {"regions": [
        {"region_id": "ZZZ", "description": "a", "source_reference": {},
         "continuation_context": {"stitch_id": "doc:s000007"}},
        {"region_id": "QQQ", "description": "b", "source_reference": {},
         "continuation_context": {"stitch_id": "doc:s999999"}},
    ]}
    page = PageData("doc", 3, "x.png", "md", "html")
    ra = region_analyzer._normalize(raw, page, carry)
    assert [r.region_id for r in ra.regions] == ["r001", "r002"]           # ids reassigned
    assert ra.regions[0].continuation_context == {"stitch_id": "doc:s000007"}  # valid carry stitch kept
    assert ra.regions[1].continuation_context is None                     # invented stitch dropped


def test_route_regions_parses_per_region(monkeypatch):
    from src.llm import structured_output as so
    from src.registry.skill_registry import get_registry

    canned = json.dumps({"page_number": 3, "regions": [
        {"region_id": "r001",
         "skills": [{"skill_id": "table", "version": 1, "relevance": 0.9, "reason": "grid"}],
         "coverage": "sufficient"}]})
    monkeypatch.setattr(so, "chat_completion", lambda **k: canned)

    tbl = get_registry().get("table")
    ra = RegionAnalysis.from_dict({"page_number": 3, "regions": [
        {"region_id": "r001", "description": "a duty table", "source_reference": {},
         "continuation_context": None}]})
    page = PageData("doc", 3, "x.png", "md", "html")
    sel = skill_router.route_regions(page, CarryState.empty(), ra, {"r001": [tbl]}, image=None)
    assert sel.regions[0].region_id == "r001"
    assert [s.skill_id for s in sel.regions[0].skills] == ["table"]

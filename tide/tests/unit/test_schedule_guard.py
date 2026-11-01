"""Unit tests for the P2 schedule over-segmentation safety nets (items 4b + 6).

All deterministic — no LLM. Covers:
  - the numbering primitives (`_leading_ordinal`, `is_plausible_successor`) in carry_rejection,
  - the uniform-run detector + re-nest collapse in schedule_guard (text preserved byte-for-byte),
  - the carry collision guard through `analyze_carry` and through the live orchestrator
    (`_resolve_continuation`, with the judge monkeypatched),
  - the item-5 retrieval reinforcement (a serial-list query surfaces the `table` skill).
"""

from src.contracts.continuation import CarryState, ContinuationDecision
from src.contracts.page import PageData, PageExtraction
from src.core import config
from src.stages import carry_rejection as cr
from src.stages import schedule_guard as sg


# --------------------------------------------------------------------------- primitives
def test_leading_ordinal_bangla_and_ascii():
    assert cr._leading_ordinal("১০। তফসিল") == 10
    assert cr._leading_ordinal("3. Something") == 3
    assert cr._leading_ordinal("Section 4") is None      # no leading numeral+danda/dot
    assert cr._leading_ordinal("(১) sub-point") is None  # bracketed, not a main rule
    assert cr._leading_ordinal("২০২৪ gazette") is None   # bare number, no danda/dot
    assert cr._leading_ordinal(None) is None


def test_is_plausible_successor():
    assert cr.is_plausible_successor("১০।", "১১। next rule") is True   # 11 >= 10
    assert cr.is_plausible_successor("10.", "10. same") is True        # equal ok
    assert cr.is_plausible_successor("১০।", "২। জাতীয়তা") is False     # field-2 collision vs expect >=10
    assert cr.is_plausible_successor(None, "২। জাতীয়তা") is True       # no expectation -> fall back
    assert cr.is_plausible_successor("১০।", "not numbered") is True    # unparseable fresh -> fall back


# --------------------------------------------------------------------------- detector
def _pb(*rows):
    """rows are (block_id, text) top-level null-stitch text_blocks unless a 5-tuple is given."""
    out = []
    for r in rows:
        if len(r) == 5:
            out.append(r)
        else:
            bid, text = r
            out.append((bid, "text_block", None, None, text))
    return out


def test_detect_fires_on_uniform_run():
    pb = _pb(("b1", "১। জাতীয়তা"), ("b2", "২। পিতার নাম"),
             ("b3", "৩। ঠিকানা"), ("b4", "৪। স্বাক্ষর"))
    runs = sg.detect_uniform_numbered_runs(pb, min_run=4, max_item_chars=160)
    assert len(runs) == 1
    assert runs[0]["block_ids"] == ["b1", "b2", "b3", "b4"]
    assert runs[0]["parent_kind"] == "self" and runs[0]["owning_parent_id"] == "b1"


def test_detect_respects_min_run():
    pb = _pb(("b1", "১। a"), ("b2", "২। b"), ("b3", "৩। c"))  # only 3, min_run=4
    assert sg.detect_uniform_numbered_runs(pb, min_run=4, max_item_chars=160) == []


def test_detect_excludes_long_items():
    pb = _pb(*[(f"b{i}", f"{i}. " + "x" * 200) for i in range(1, 6)])
    assert sg.detect_uniform_numbered_runs(pb, min_run=4, max_item_chars=160) == []


def test_detect_excludes_blocks_with_children():
    # b2 has a child c2, so b2 is a real rule (referenced as a parent) -> not a shred item;
    # the run is broken and no >=4 run survives.
    pb = _pb(("b1", "text_block", None, None, "১। a"),
             ("b2", "text_block", None, None, "২। b"),
             ("c2", "text_block", "b2", None, "(ক) child"),
             ("b3", "text_block", None, None, "৩। c"),
             ("b4", "text_block", None, None, "৪। d"))
    assert sg.detect_uniform_numbered_runs(pb, min_run=4, max_item_chars=160) == []


def test_detect_owner_is_preceding_heading():
    pb = _pb(("h", "heading", None, None, "তফসিল-১"),
             ("b1", "১। a"), ("b2", "২। b"), ("b3", "৩। c"), ("b4", "৪। d"))
    runs = sg.detect_uniform_numbered_runs(pb, min_run=4, max_item_chars=160)
    assert runs[0]["parent_kind"] == "schedule_heading" and runs[0]["owning_parent_id"] == "h"


def test_analyze_over_segmentation_record_shape():
    pb = _pb(("b1", "১। a"), ("b2", "২। b"), ("b3", "৩। c"), ("b4", "৪। d"))
    rec = sg.analyze_over_segmentation(pb, min_run=4, max_item_chars=160)
    assert rec["run_count"] == 1 and rec["flagged_block_count"] == 4
    assert rec["min_run"] == 4 and rec["max_item_chars"] == 160


# --------------------------------------------------------------------------- collapse
def _extraction(rows):
    return PageExtraction.from_dict({"page_number": 1, "blocks": [
        {"block_id": bid, "block_type": bt, "parent_block_id": par,
         "stitch_id": None, "content": {"text": txt}, "skills_used": []}
        for bid, bt, par, txt in rows]})


def test_collapse_self_owner_preserves_text_and_renests():
    ext = _extraction([("b1", "text_block", None, "১। a"), ("b2", "text_block", None, "২। b"),
                       ("b3", "text_block", None, "৩। c"), ("b4", "text_block", None, "৪। d")])
    before = [b.content["text"] for b in ext.blocks]
    runs = sg.detect_uniform_numbered_runs(sg.page_blocks_of(ext.blocks), 4, 160)
    ext.blocks = sg.collapse_uniform_runs(ext.blocks, runs)
    assert [b.content["text"] for b in ext.blocks] == before        # text byte-for-byte
    top = [b for b in ext.blocks if b.parent_block_id is None]
    assert [b.block_id for b in top] == ["b1"]                      # over-seg collapsed 4 -> 1 top-level
    assert all(b.parent_block_id == "b1" for b in ext.blocks if b.block_id != "b1")


def test_collapse_under_schedule_heading():
    ext = _extraction([("h", "heading", None, "তফসিল-১"),
                       ("b1", "text_block", None, "১। a"), ("b2", "text_block", None, "২। b"),
                       ("b3", "text_block", None, "৩। c"), ("b4", "text_block", None, "৪। d")])
    runs = sg.detect_uniform_numbered_runs(sg.page_blocks_of(ext.blocks), 4, 160)
    sg.collapse_uniform_runs(ext.blocks, runs)
    top = [b for b in ext.blocks if b.parent_block_id is None]
    assert {b.block_id for b in top} == {"h"}                       # only the cover stays top-level
    assert all(b.parent_block_id == "h" for b in ext.blocks if b.block_type == "text_block")


# --------------------------------------------------------------------------- carry guard (ledger)
def _open_main_unit(expected="১০।"):
    return [{"stitch_id": "s10", "block_type": "text_block", "is_main_point": True,
             "expected_next_main_point": expected, "state": "", "open_text": ""}]


def test_analyze_carry_default_path_unchanged_by_collision():
    # A fresh field-2 collides with a rule-10 open unit. Historical behaviour: any fresh main
    # closes it. Default (guard off) MUST keep this to stay metric-invariant.
    page = [("x", "text_block", None, None, "২। জাতীয়তা")]
    rec = cr.analyze_carry(_open_main_unit(), page, collision_guard=False)["units"][0]
    assert rec["status"] == "closed_by_successor" and rec["symptom"] is None


def test_analyze_carry_guard_rejects_collision():
    page = [("x", "text_block", None, None, "২। জাতীয়তা")]
    rec = cr.analyze_carry(_open_main_unit(), page, collision_guard=True)["units"][0]
    assert rec["status"] == "rejected" and rec["symptom"] == "main_dropped"


def test_analyze_carry_guard_allows_real_successor():
    page = [("x", "text_block", None, None, "১১। পরবর্তী বিধি")]
    rec = cr.analyze_carry(_open_main_unit(), page, collision_guard=True)["units"][0]
    assert rec["status"] == "closed_by_successor"


# --------------------------------------------------------------------------- orchestrator wiring
def _closing_setup():
    """A page whose open main (s10, expecting ১০।) is closed while only a colliding field-2
    fresh main appears — the exact T11 shape the guard must catch."""
    carry = CarryState.from_dict({"from_page": 0, "open_units": [
        {"stitch_id": "s10", "last_block_id": "prev", "block_type": "text_block",
         "state": "open rule 10", "open_text": "rule 10 body", "continuation_evidence": [],
         "skills": [], "is_main_point": True, "expected_next_main_point": "১০।"}]})
    ext = PageExtraction.from_dict({"page_number": 1, "blocks": [
        {"block_id": "a1", "block_type": "text_block", "parent_block_id": None,
         "stitch_id": "s10", "content": {"text": "…tail of rule 10"}, "skills_used": []},
        {"block_id": "f2", "block_type": "text_block", "parent_block_id": None,
         "stitch_id": None, "content": {"text": "২। জাতীয়তা"}, "skills_used": []}],
        "continuation": {"page_number": 1, "decisions": [
            {"block_id": "a1", "stitch_id": "s10", "is_main_point": False, "decision": "ends",
             "evidence": []},
            {"block_id": "f2", "stitch_id": None, "is_main_point": True, "decision": "continues",
             "evidence": []}]}})
    return ext, carry


def test_resolve_continuation_guard_flags_collision(monkeypatch):
    from src.core import orchestrator
    monkeypatch.setattr(config, "CARRY_COLLISION_GUARD", True)
    # Judge just echoes the flagged block back as "ends" so we can observe the routing.
    monkeypatch.setattr(orchestrator.continuation_judge, "judge",
                        lambda *a, **k: ContinuationDecision(1, []))
    ext, carry = _closing_setup()
    page, nxt = PageData("doc", 1, "p1.png", "md", "h"), PageData("doc", 2, "p2.png", "md", "h")
    _, rec = orchestrator._resolve_continuation(ext, page, nxt, carry, None, "coerce",
                                                judge_vision=False)
    assert "a1" in rec["b2_flagged"] and rec["judge_invoked"] is True


def test_resolve_continuation_guard_off_does_not_flag(monkeypatch):
    from src.core import orchestrator
    monkeypatch.setattr(config, "CARRY_COLLISION_GUARD", False)
    monkeypatch.setattr(orchestrator.continuation_judge, "judge",
                        lambda *a, **k: ContinuationDecision(1, []))
    ext, carry = _closing_setup()
    page, nxt = PageData("doc", 1, "p1.png", "md", "h"), PageData("doc", 2, "p2.png", "md", "h")
    _, rec = orchestrator._resolve_continuation(ext, page, nxt, carry, None, "coerce",
                                                judge_vision=False)
    assert rec["b2_flagged"] == []  # a fresh main is present, so the close stands (historical)


# --------------------------------------------------------------------------- item 5 retrieval
def test_serial_list_query_surfaces_table_skill(monkeypatch):
    from src.registry.skill_retriever import retrieve_skills
    monkeypatch.setattr(config, "RETRIEVAL_SCORER", "keyword")  # stdlib scorer, no ST dependency
    q = "a long uniform parallel-structured serial appendix schedule catalogue list of numbered items"
    result = retrieve_skills(q, ["main"], config.MAIN_TOP_K)
    assert "table" in {r.skill_id for r in result}

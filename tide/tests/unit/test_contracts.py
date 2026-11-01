from src.contracts.page import PageExtraction
from src.contracts.continuation import CarryState
from src.contracts.skill import SkillRecord


def test_page_extraction_round_trip():
    d = {
        "page_number": 3,
        "blocks": [
            {"block_id": "doc:p0003:b0001", "block_type": "heading",
             "parent_block_id": None, "stitch_id": None,
             "content": {"text": "H"}, "skills_used": []},
            {"block_id": "doc:p0003:b0002", "block_type": "text_block",
             "parent_block_id": None, "stitch_id": "doc:s000001",
             "content": {"text": "T"},
             "skills_used": [{"skill_id": "numbered_point", "version": 1}]},
        ],
        "continuation": {"page_number": 3, "decisions": [
            {"block_id": "doc:p0003:b0002", "stitch_id": "doc:s000001",
             "decision": "continues", "evidence": ["x"]}]},
    }
    pe = PageExtraction.from_dict(d)
    assert pe.to_dict() == d


def test_carry_state_round_trip_and_empty():
    assert CarryState.empty(5).to_dict() == {"from_page": 5, "open_units": []}
    d = {"from_page": 2, "open_units": [
        {"stitch_id": "doc:s000001", "last_block_id": "doc:p0002:b0001",
         "block_type": "text_block", "state": "open",
         "continuation_evidence": ["e"], "skills": [{"skill_id": "x", "version": 1}]}]}
    assert CarryState.from_dict(d).to_dict() == d


def test_skill_record_extractor_view_hides_provenance():
    rec = SkillRecord.from_dict({
        "skill_id": "table", "version": 3, "status": "main",
        "description": "d", "applicability": ["a"], "instructions": ["i1", "i2"],
        "conflicts_with": [], "provenance": {"reason": "secret"},
    })
    view = rec.extractor_view()
    assert view == {"skill_id": "table", "version": 3, "instructions": ["i1", "i2"]}
    assert "provenance" not in view

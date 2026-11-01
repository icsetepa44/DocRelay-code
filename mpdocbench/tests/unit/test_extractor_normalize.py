from src.contracts.page import PageData
from src.contracts.continuation import CarryState
from src.stages import extractor


def test_normalize_ids_parents_types_and_stitch_reuse():
    carry = CarryState.from_dict({"from_page": 0, "open_units": [
        {"stitch_id": "doc:s000009", "last_block_id": "doc:p0000:b0002",
         "block_type": "text_block", "state": "open",
         "continuation_evidence": [], "skills": []}]})
    raw = {
        "page_number": 1,
        "blocks": [
            {"block_id": "h", "block_type": "heading", "parent_block_id": None,
             "stitch_id": None, "content": {"text": "Head"}, "skills_used": []},
            {"block_id": "p1", "block_type": "text_block", "parent_block_id": None,
             "stitch_id": "doc:s000009", "content": {"text": "cont"},
             "skills_used": [{"skill_id": "numbered_point", "version": 1}]},
            {"block_id": "sub", "block_type": "NONSENSE", "parent_block_id": "p1",
             "stitch_id": "doc:s999999", "content": {"text": "child"}, "skills_used": []},
        ],
        "continuation": {"page_number": 1, "decisions": [
            {"block_id": "sub", "decision": "continues", "evidence": ["e"]},
            {"block_id": "ghost", "decision": "continues", "evidence": []},  # unknown -> dropped
        ]},
    }
    page = PageData("doc", 1, "x.png", "md", "html")
    pe = extractor.normalize_extraction(raw, page, carry)

    assert [b.block_id for b in pe.blocks] == [
        "doc:p0001:b0001", "doc:p0001:b0002", "doc:p0001:b0003"]
    assert pe.blocks[2].parent_block_id == "doc:p0001:b0002"   # local parent remapped
    assert pe.blocks[2].block_type == "text_block"             # invalid type coerced
    assert pe.blocks[1].stitch_id == "doc:s000009"             # reused from carry
    assert pe.blocks[2].stitch_id is None                      # invented stitch dropped
    # only the known-block decision survives, remapped to canonical id
    assert [d.block_id for d in pe.continuation.decisions] == ["doc:p0001:b0003"]

from src.contracts.page import PageExtraction
from src.core import config
from src.core.ids import StitchIdAllocator
from src.stages import carry_builder


def _extraction():
    return PageExtraction.from_dict({
        "page_number": 1,
        "blocks": [
            {"block_id": "doc:p0001:b0001", "block_type": "text_block",
             "parent_block_id": None, "stitch_id": None,
             "content": {"text": "ends here."}, "skills_used": []},
            {"block_id": "doc:p0001:b0002", "block_type": "text_block",
             "parent_block_id": None, "stitch_id": None,
             "content": {"text": "this point continues"}, "skills_used": []},
        ],
        "continuation": {"page_number": 1, "decisions": [
            {"block_id": "doc:p0001:b0001", "decision": "ends", "evidence": []},
            {"block_id": "doc:p0001:b0002", "decision": "continues", "evidence": ["incomplete"]},
        ]},
    })


def test_allocates_new_stitch_for_continuing_block():
    pe = _extraction()
    alloc = StitchIdAllocator("doc")
    cs = carry_builder.build_carry_state(pe, pe.continuation, alloc)
    assert pe.blocks[1].stitch_id == "doc:s000001"
    assert pe.blocks[0].stitch_id is None
    assert [u.stitch_id for u in cs.open_units] == ["doc:s000001"]
    assert cs.open_units[0].last_block_id == "doc:p0001:b0002"
    assert cs.from_page == 1


def test_reuses_existing_stitch_id():
    pe = _extraction()
    pe.blocks[1].stitch_id = "doc:s000009"
    alloc = StitchIdAllocator("doc", start=9)
    cs = carry_builder.build_carry_state(pe, pe.continuation, alloc)
    assert pe.blocks[1].stitch_id == "doc:s000009"
    assert alloc.issued == 9
    assert cs.open_units[0].stitch_id == "doc:s000009"


def _table_extraction():
    return PageExtraction.from_dict({
        "page_number": 1,
        "blocks": [
            {"block_id": "doc:p0001:b0001", "block_type": "table",
             "parent_block_id": None, "stitch_id": None,
             "content": {"table": {"type": "generic_table",
                                   "headers": ["ক্রমিক", "পণ্য"],
                                   "rows": [["১", "চাল"], ["২", "ডাল"]]},
                         "parsing_status": "success"},
             "skills_used": []},
        ],
        "continuation": {"page_number": 1, "decisions": [
            {"block_id": "doc:p0001:b0001", "decision": "continues",
             "evidence": ["table continues"]},
        ]},
    })


def test_open_table_carries_real_state_and_open_text():
    pe = _table_extraction()
    alloc = StitchIdAllocator("doc")
    cs = carry_builder.build_carry_state(pe, pe.continuation, alloc)
    u = cs.open_units[0]
    assert u.block_type == "table"
    assert "continues onto the next page" not in u.state
    assert "ক্রমিক" in u.state and "চাল" in u.state
    if config.CARRY_OPEN_TEXT_MAX_CHARS > 0:
        assert u.open_text and "পণ্য" in u.open_text and "ডাল" in u.open_text

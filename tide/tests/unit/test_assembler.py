from src.contracts.page import PageExtraction
from src.stages import assembler


def test_stitched_fragments_merge_and_parent_links():
    p0 = PageExtraction.from_dict({"page_number": 0, "blocks": [
        {"block_id": "doc:p0000:b0001", "block_type": "heading",
         "parent_block_id": None, "stitch_id": None,
         "content": {"text": "Title"}, "skills_used": []},
        {"block_id": "doc:p0000:b0002", "block_type": "text_block",
         "parent_block_id": None, "stitch_id": "doc:s000001",
         "content": {"text": "point start"}, "skills_used": []},
    ]})
    p1 = PageExtraction.from_dict({"page_number": 1, "blocks": [
        {"block_id": "doc:p0001:b0001", "block_type": "text_block",
         "parent_block_id": None, "stitch_id": "doc:s000001",
         "content": {"text": "point end"}, "skills_used": []},
        {"block_id": "doc:p0001:b0002", "block_type": "table",
         "parent_block_id": "doc:p0001:b0001", "stitch_id": None,
         "content": {"markdown": "|a|", "parsing_status": "fallback"}, "skills_used": []},
    ]})

    fd = assembler.assemble("doc", [p0, p1])
    ids_in_order = [u.unit_id for u in fd.units]
    assert ids_in_order == ["doc:p0000:b0001", "doc:s000001", "doc:p0001:b0002"]

    units = {u.unit_id: u for u in fd.units}
    merged = units["doc:s000001"]
    assert isinstance(merged.content, list) and len(merged.content) == 2  # no flatten
    assert merged.source_block_ids == ["doc:p0000:b0002", "doc:p0001:b0001"]
    # child table's parent unit is the stitched point (parent-child preserved)
    assert units["doc:p0001:b0002"].parent_unit_id == "doc:s000001"


def test_standalone_block_stays_single_content():
    p = PageExtraction.from_dict({"page_number": 0, "blocks": [
        {"block_id": "doc:p0000:b0001", "block_type": "text_block",
         "parent_block_id": None, "stitch_id": None,
         "content": {"text": "solo"}, "skills_used": []},
    ]})
    fd = assembler.assemble("doc", [p])
    assert fd.units[0].content == {"text": "solo"}
    assert fd.units[0].unit_id == "doc:p0000:b0001"

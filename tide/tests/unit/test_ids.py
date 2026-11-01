from src.core import ids


def test_block_id_format():
    assert ids.block_id("doc_001", 12, 4) == "doc_001:p0012:b0004"


def test_stitch_id_format():
    assert ids.stitch_id("doc_001", 7) == "doc_001:s000007"


def test_failure_id_format():
    assert ids.failure_id("doc_001", 18) == "doc_001:failure:000018"


def test_unit_id_for():
    assert ids.unit_id_for("doc:p0001:b0001", None) == "doc:p0001:b0001"
    assert ids.unit_id_for("doc:p0001:b0001", "doc:s000007") == "doc:s000007"


def test_stitch_allocator_is_monotonic():
    alloc = ids.StitchIdAllocator("doc")
    assert alloc.next() == "doc:s000001"
    assert alloc.next() == "doc:s000002"
    assert alloc.issued == 2

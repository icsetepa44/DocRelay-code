import json

from src.contracts.page import PageData
from src.core import resume
from src.storage import run_store

DOC = "doc"
LABEL = "modelL"


def _block(page: int, seq: int, stitch: str | None):
    return {
        "block_id": f"{DOC}:p{page:04d}:b{seq:04d}",
        "block_type": "text_block",
        "parent_block_id": None,
        "stitch_id": stitch,
        "content": {"text": "x"},
        "skills_used": [],
    }


def _open_unit(stitch: str, last_block_id: str):
    return {"stitch_id": stitch, "last_block_id": last_block_id,
            "block_type": "text_block", "state": "open"}


def _write_page(run_dir, page_number, *, blocks, carry, torn=False):
    pdir = run_store.page_dir(run_dir, DOC, page_number, LABEL)
    extraction = {"page_number": page_number, "blocks": blocks, "continuation": None}
    (pdir / "extraction.json").write_text(json.dumps(extraction), encoding="utf-8")
    if not torn:
        (pdir / "carry_state.json").write_text(json.dumps(carry), encoding="utf-8")
    return pdir


def _pages(n):
    return [PageData(document_id=DOC, page_number=i, page_image="",
                     markdown="some body text", html="") for i in range(n)]


def _partial_run(run_dir):
    _write_page(run_dir, 0,
                blocks=[_block(0, 1, f"{DOC}:s000001")],
                carry={"from_page": 0,
                       "open_units": [_open_unit(f"{DOC}:s000001", f"{DOC}:p0000:b0001")]})
    _write_page(run_dir, 1,
                blocks=[_block(1, 1, f"{DOC}:s000001"), _block(1, 2, f"{DOC}:s000002")],
                carry={"from_page": 1,
                       "open_units": [_open_unit(f"{DOC}:s000002", f"{DOC}:p0001:b0002")]})
    _write_page(run_dir, 2, blocks=[_block(2, 1, None)], carry={}, torn=True)


def test_page_complete_true_for_valid_pair(tmp_path):
    pdir = _write_page(tmp_path, 0, blocks=[_block(0, 1, None)],
                       carry={"from_page": 0, "open_units": []})
    assert resume.page_complete(pdir) is True


def test_page_complete_false_when_carry_missing(tmp_path):
    pdir = _write_page(tmp_path, 0, blocks=[_block(0, 1, None)], carry={}, torn=True)
    assert resume.page_complete(pdir) is False


def test_page_complete_false_on_torn_json(tmp_path):
    pdir = run_store.page_dir(tmp_path, DOC, 0, LABEL)
    (pdir / "extraction.json").write_text('{"page_number": 0, "blocks": [', encoding="utf-8")
    (pdir / "carry_state.json").write_text("{}", encoding="utf-8")
    assert resume.page_complete(pdir) is False


def test_page_complete_false_for_missing_dir(tmp_path):
    assert resume.page_complete(tmp_path / "nope" / "pages" / "0000") is False


def test_resume_prefix_stops_at_torn_page(tmp_path):
    run_dir = tmp_path / "run_x"
    _partial_run(run_dir)
    rs = resume.compute_resume_state(run_dir, DOC, LABEL, _pages(3), min_chars_nontrivial=1)

    assert rs.index == 2
    assert rs.carry.from_page == 1
    assert [u.stitch_id for u in rs.carry.open_units] == [f"{DOC}:s000002"]
    assert rs.allocator_start == 2
    assert [e.page_number for e in rs.extractions] == [0, 1]
    assert rs.resume_page_number == 2
    assert [s["page_number"] for s in rs.page_summaries] == [0, 1]
    assert rs.page_summaries[1]["open_units_after"] == [f"{DOC}:s000002"]


def test_clean_incomplete_tail_removes_torn_page(tmp_path):
    run_dir = tmp_path / "run_x"
    _partial_run(run_dir)
    removed = resume.clean_incomplete_tail(run_dir, DOC, LABEL, _pages(3), 2)
    assert removed == [2]
    assert not run_store.page_dir_path(run_dir, DOC, 2, LABEL).exists()
    assert run_store.page_dir_path(run_dir, DOC, 0, LABEL).exists()
    assert run_store.page_dir_path(run_dir, DOC, 1, LABEL).exists()


def test_resume_index_zero_is_fresh(tmp_path):
    run_dir = tmp_path / "run_fresh"
    run_dir.mkdir()
    rs = resume.compute_resume_state(run_dir, DOC, LABEL, _pages(3), min_chars_nontrivial=1)
    assert rs.index == 0
    assert rs.extractions == []
    assert rs.carry.open_units == []
    assert rs.allocator_start == 0
    assert rs.resume_page_number == 0
    assert resume.clean_incomplete_tail(run_dir, DOC, LABEL, _pages(3), 0) == []


def test_resume_prefix_stops_at_gap(tmp_path):
    run_dir = tmp_path / "run_gap"
    _write_page(run_dir, 0, blocks=[_block(0, 1, f"{DOC}:s000001")],
                carry={"from_page": 0, "open_units": []})
    _write_page(run_dir, 1, blocks=[_block(1, 1, f"{DOC}:s000002")],
                carry={"from_page": 1, "open_units": []})
    _write_page(run_dir, 3, blocks=[_block(3, 1, f"{DOC}:s000009")],
                carry={"from_page": 3, "open_units": []})
    rs = resume.compute_resume_state(run_dir, DOC, LABEL, _pages(4), min_chars_nontrivial=1)
    assert rs.index == 2
    assert rs.resume_page_number == 2
    assert resume.clean_incomplete_tail(run_dir, DOC, LABEL, _pages(4), 2) == [3]


def test_resume_whole_slice_complete(tmp_path):
    run_dir = tmp_path / "run_all"
    _write_page(run_dir, 0, blocks=[_block(0, 1, f"{DOC}:s000001")],
                carry={"from_page": 0, "open_units": []})
    _write_page(run_dir, 1, blocks=[_block(1, 1, f"{DOC}:s000002")],
                carry={"from_page": 1, "open_units": []})
    rs = resume.compute_resume_state(run_dir, DOC, LABEL, _pages(2), min_chars_nontrivial=1)
    assert rs.index == 2
    assert rs.resume_page_number is None
    assert rs.allocator_start == 2
    assert resume.clean_incomplete_tail(run_dir, DOC, LABEL, _pages(2), 2) == []

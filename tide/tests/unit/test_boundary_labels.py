"""Tests for the per-page-boundary split/merge labels (src/eval/boundary_labels.py).

Synthetic gold + final_document.json + pages/ on tmp_path. Gold pages are 1-based; pipeline block ids /
pages/ dirs are 0-based (gold page p == pipeline page p-1)."""

import json

from src.eval import boundary_labels as bl

DOC = "doc"


def _bid(page0, seq):
    return f"{DOC}:p{page0:04d}:b{seq:04d}"


def _unit(uid, text, sources, parent=None, bt="text_block"):
    return {"unit_id": uid, "block_type": bt, "content": {"text": text}, "parent_unit_id": parent,
            "source_block_ids": list(sources)}


def _clause(num, text, ps, pe, typ="clause"):
    return {"type": typ, "clause_number": num, "text": text, "page_start": ps, "page_end": pe}


def _setup(tmp_path, units, n_pages, page_files=None, missing=()):
    """Write final_document.json + pages/<p>/extraction.json (+ optional extra per-page files)."""
    d = tmp_path / DOC / "m"
    for p in range(n_pages):
        if p in missing:
            continue
        pd = d / "pages" / f"{p:04d}"
        pd.mkdir(parents=True)
        files = {"extraction.json": {"page_number": p, "blocks": [], "continuation": {"decisions": []}},
                 "carry_state.json": {"from_page": p, "open_units": []},
                 "carry_rejection.json": {"page_number": p, "units": [], "rejected_count": 0}}
        files.update((page_files or {}).get(p, {}))
        for name, obj in files.items():
            (pd / name).write_text(json.dumps(obj), encoding="utf-8")
    d.mkdir(parents=True, exist_ok=True)
    fd = d / "final_document.json"
    fd.write_text(json.dumps({"units": units}), encoding="utf-8")
    return fd


def _gold(items, pages):
    return {"document_info": {"total_pages": pages}, "parsed_text": items, "tables": []}


def _outcomes(res):
    return [(r["page_before"], r["outcome"]) for r in res["rows"]]


A = "alpha beta gamma delta epsilon zeta eta theta"
B = "iota kappa lambda mu nu xi omicron pi"


def test_carried_when_one_predicted_clause_spans_the_boundary(tmp_path):
    fd = _setup(tmp_path, [_unit("s1", A, [_bid(0, 1), _bid(1, 1)])], 2)
    res = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)
    assert _outcomes(res) == [(1, "carried")]
    assert res["rows"][0]["overlap"] == 1.0 and not res["rows"][0]["suspect"]


def test_split_never_opened(tmp_path):
    fd = _setup(tmp_path, [_unit(_bid(0, 1), "alpha beta gamma delta", [_bid(0, 1)]),
                           _unit(_bid(1, 1), "epsilon zeta eta theta", [_bid(1, 1)])], 2)
    res = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)
    assert _outcomes(res) == [(1, "split")]
    assert res["rows"][0]["cause"] == "never_opened"
    assert res["rows"][0]["gold_clauses"] == "c1" and res["rows"][0]["gold_span"] == 1


def test_split_opened_rejected_keeps_symptom(tmp_path):
    files = {0: {"carry_state.json": {"from_page": 0, "open_units": [{"stitch_id": f"{DOC}:s000001"}]}},
             1: {"carry_rejection.json": {"page_number": 1, "rejected_count": 1, "units": [
                 {"stitch_id": f"{DOC}:s000001", "status": "rejected", "symptom": "promoted_subpoint"}]}}}
    fd = _setup(tmp_path, [_unit(_bid(0, 1), "a b c", [_bid(0, 1)]), _unit(_bid(1, 1), "d e f", [_bid(1, 1)])],
                2, page_files=files)
    row = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)["rows"][0]
    assert (row["outcome"], row["cause"], row["carry_symptom"]) == ("split", "opened_rejected", "promoted_subpoint")


def test_split_opened_closed_and_other(tmp_path):
    files = {0: {"carry_state.json": {"from_page": 0, "open_units": [{"stitch_id": "x"}]}},
             1: {"carry_rejection.json": {"units": [{"stitch_id": "x", "status": "closed_by_successor"}]}}}
    fd = _setup(tmp_path / "a", [_unit(_bid(0, 1), "a", [_bid(0, 1)]), _unit(_bid(1, 1), "b", [_bid(1, 1)])],
                2, page_files=files)
    row = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)["rows"][0]
    assert (row["cause"], row["carry_symptom"]) == ("opened_closed", "closed_by_successor")

    files = {0: {"extraction.json": {"blocks": [], "continuation": {"decisions": [{"decision": "continues"}]}}}}
    fd = _setup(tmp_path / "b", [_unit(_bid(0, 1), "a", [_bid(0, 1)]), _unit(_bid(1, 1), "b", [_bid(1, 1)])],
                2, page_files=files)
    assert bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)["rows"][0]["cause"] == "other"


def test_joined_under_metadata_item_is_carried_and_flagged(tmp_path):
    """A stitched unit crosses the page under a heading (metadata top item): the pages were kept together,
    so it is `carried` (a typing error, not a split) and flagged joined_as_metadata."""
    units = [_unit("h", "Schedule", [_bid(0, 1)], bt="heading"),
             _unit("s1", A, [_bid(0, 2), _bid(1, 1)], parent="h")]
    fd = _setup(tmp_path, units, 2)
    row = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)["rows"][0]
    assert (row["outcome"], row["joined_as_metadata"]) == ("carried", True)


def test_split_stitched_but_dropped(tmp_path):
    """A stitched unit crosses the page, but its top item is noise_or_separator, which adapt drops."""
    units = [_unit("n", "----", [_bid(0, 1)], bt="noise_or_separator"),
             _unit("s1", A, [_bid(0, 2), _bid(1, 1)], parent="n")]
    fd = _setup(tmp_path, units, 2)
    row = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)["rows"][0]
    assert (row["outcome"], row["cause"]) == ("split", "stitched_but_dropped")


def test_split_cause_unknown_when_a_page_file_is_missing(tmp_path):
    fd = _setup(tmp_path, [_unit(_bid(0, 1), "a", [_bid(0, 1)]), _unit(_bid(1, 1), "b", [_bid(1, 1)])], 2)
    (fd.parent / "pages" / "0001" / "carry_rejection.json").unlink()
    assert bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 2)], 2), 2)["rows"][0]["cause"] == "unknown"


def test_merge_via_stitch_and_via_parent_and_overlap(tmp_path):
    gold = _gold([_clause("c1", A, 1, 1), _clause("c2", B, 2, 2)], 2)
    fd = _setup(tmp_path / "s", [_unit("s1", A + " " + B, [_bid(0, 1), _bid(1, 1)])], 2)
    row = bl.label_document(DOC, "m", fd, gold, 2)["rows"][0]
    assert (row["outcome"], row["cause"], row["overlap"], row["suspect"]) == ("merge", "via_stitch", 1.0, False)
    assert row["gold_clauses"] == "c1;c2"

    fd = _setup(tmp_path / "p", [_unit(_bid(0, 1), A, [_bid(0, 1)]),
                                 _unit(_bid(1, 1), B, [_bid(1, 1)], parent=_bid(0, 1))], 2)
    row = bl.label_document(DOC, "m", fd, gold, 2)["rows"][0]
    assert (row["outcome"], row["cause"]) == ("merge", "via_parent")


def test_correct_break_and_suspect_carry(tmp_path):
    gold = _gold([_clause("c1", A, 1, 1), _clause("c2", B, 2, 3)], 3)
    # p1|2: two separate items -> correct_break; p2|3: joined, but with unrelated text -> suspect carried
    fd = _setup(tmp_path, [_unit(_bid(0, 1), A, [_bid(0, 1)]),
                           _unit("s1", "totally different words here", [_bid(1, 1), _bid(2, 1)])], 3)
    res = bl.label_document(DOC, "m", fd, gold, 3)
    assert _outcomes(res) == [(1, "correct_break"), (2, "carried")]
    assert res["rows"][1]["suspect"] and res["rows"][1]["overlap"] == 0.0


def test_multi_page_span_labels_every_interior_boundary(tmp_path):
    """A gold clause on pages 1-4 = 3 continues-boundaries; the pipeline breaks only at 2|3."""
    fd = _setup(tmp_path, [_unit("s1", "a b", [_bid(0, 1), _bid(1, 1)]),
                           _unit("s2", "c d", [_bid(2, 1), _bid(3, 1)])], 4)
    res = bl.label_document(DOC, "m", fd, _gold([_clause("c1", "a b c d", 1, 4)], 4), 4)
    assert _outcomes(res) == [(1, "carried"), (2, "split"), (3, "carried")]
    assert {r["gold_span"] for r in res["rows"]} == {3}


def test_page_missing_and_page_count_mismatch_and_single_page(tmp_path):
    fd = _setup(tmp_path / "m", [_unit("s1", A, [_bid(0, 1), _bid(2, 1)])], 3, missing={1})
    res = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 3)], 3), 3)
    assert _outcomes(res) == [(1, "page_missing"), (2, "page_missing")]

    res = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 3)], 3), 5)
    assert res["skip_reason"] == "page_count_mismatch" and res["rows"] == []

    fd1 = _setup(tmp_path / "one", [_unit(_bid(0, 1), A, [_bid(0, 1)])], 1)
    assert bl.label_document(DOC, "m", fd1, _gold([_clause("c1", A, 1, 1)], 1), 1)["skip_reason"] == "single_page"


def test_metadata_span_excluded_and_page_metadata_ignored(tmp_path):
    items = [_clause("m1", "preamble words", 1, 2, typ="metadata"),
             {**_clause("pg", "2", 1, 2, typ="metadata"), "metadata_category": "page"}]
    fd = _setup(tmp_path, [_unit("s1", "preamble words", [_bid(0, 1), _bid(1, 1)])], 2)
    assert _outcomes(bl.label_document(DOC, "m", fd, _gold(items, 2), 2)) == [(1, "metadata_span")]
    assert [x["number"] for x in bl.gold_items(_gold(items, 2))] == ["m1"]


def test_table_placeholder_expanded_for_overlap(tmp_path):
    gold = {"document_info": {"total_pages": 2}, "tables": [{"table_id": "table_1", "headers": ["H"],
                                                             "rows": [["rice"], ["wheat"]]}],
            "parsed_text": [_clause("c1", "[table_1]", 1, 2)]}
    fd = _setup(tmp_path, [{"unit_id": "s1", "block_type": "table", "parent_unit_id": None,
                            "source_block_ids": [_bid(0, 1), _bid(1, 1)],
                            "content": {"table": {"headers": ["H"], "rows": [["rice"], ["wheat"]]}}}], 2)
    row = bl.label_document(DOC, "m", fd, gold, 2)["rows"][0]
    assert row["outcome"] == "carried" and row["overlap"] == 1.0


def test_gold_gap_when_a_page_has_no_gold_item(tmp_path):
    """Gold annotates only pages 1 and 3: boundaries touching page 2 are gold_gap, never a merge."""
    fd = _setup(tmp_path, [_unit("s1", A + " " + B, [_bid(0, 1), _bid(1, 1), _bid(2, 1)])], 3)
    res = bl.label_document(DOC, "m", fd, _gold([_clause("c1", A, 1, 1), _clause("c2", B, 3, 3)], 3), 3)
    assert _outcomes(res) == [(1, "gold_gap"), (2, "gold_gap")]
    assert {r["gold_label"] for r in res["rows"]} == {"no_gold"}


def test_gold_table_spanning_the_boundary_counts_as_continues(tmp_path):
    """No parsed_text item covers pages 2-3, but its table does: the clause continues there."""
    gold = {"document_info": {"total_pages": 3}, "parsed_text": [_clause("c1", "[table_1]", 1, 1)],
            "tables": [{"table_id": "table_1", "page_start": 1, "page_end": 3, "headers": ["H"],
                        "rows": [["rice"]]}]}
    fd = _setup(tmp_path, [_unit("s1", "rice", [_bid(0, 1), _bid(1, 1), _bid(2, 1)], bt="table")], 3)
    res = bl.label_document(DOC, "m", fd, gold, 3)
    assert _outcomes(res) == [(1, "carried"), (2, "carried")]
    assert res["rows"][1]["gold_clauses"] == "table_1"

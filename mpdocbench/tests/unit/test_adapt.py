"""Tests for the eval bridge (src/eval/adapt.py) — the additive structured `tables[]` slot
(T3) and the shared table helpers. The key invariant: `contents` is UNCHANGED (tables are
still inlined into the parent clause text), so existing metrics are unaffected; `tables[]`
is purely additional structure for downstream consumers."""

from types import SimpleNamespace

from src.eval.adapt import flatten_final_document
from src.stages._common import block_text, table_struct, table_to_text


# --- shared helpers -------------------------------------------------------------------

def test_table_to_text_flattens_headers_and_rows():
    content = {"table": {"headers": ["H1", "H2"], "rows": [["a", "b"], ["c", "d"]]}}
    assert table_to_text(content) == "H1 H2 a b c d"


def test_table_to_text_preserves_empty_cells():
    """Empty cells are joined verbatim (extra space), NOT filtered out — this keeps the eval
    bridge's inlined text byte-identical to the pre-refactor flatten so metrics can't move.
    A form row like `(ক) | ` (empty 2nd column) must render `(ক) ` not collapse the gap."""
    content = {"table": {"headers": ["Name", "Date"],
                         "rows": [["(a)", ""], ["(b)", ""]]}}
    # empty 2nd cells -> double spaces between items, exactly as the old adapt produced.
    assert table_to_text(content) == "Name Date (a)  (b)"


def test_block_text_prefers_text_then_falls_back_to_table():
    """block_text uses text/description when present, else flattens table cells (T6) — so a
    continuing table carries real content across a page break instead of ''."""
    assert block_text(SimpleNamespace(content={"text": "hello"})) == "hello"
    assert block_text(SimpleNamespace(content={"description": "a seal"})) == "a seal"
    tbl = SimpleNamespace(content={"table": {"headers": ["H"], "rows": [["x"], ["y"]]}})
    assert block_text(tbl) == "H x y"
    assert block_text(SimpleNamespace(content={})) == ""
    assert block_text(SimpleNamespace(content=None)) == ""


def test_table_to_text_fallback_to_markdown_html():
    assert table_to_text({"markdown": "| a | b |"}) == "| a | b |"
    assert table_to_text({"html": "<table></table>"}) == "<table></table>"
    assert table_to_text({}) == ""


def test_table_struct_returns_none_for_fallback():
    assert table_struct({"table": {"headers": ["H"], "rows": [["x"]]}}) == {
        "headers": ["H"], "rows": [["x"]]}
    assert table_struct({"markdown": "x"}) is None
    assert table_struct({}) is None


# --- flatten_final_document: additive tables[] slot -----------------------------------

def test_child_table_inlined_and_also_structured():
    """A table that is a child of a clause: cells still inlined in the clause text (metric
    invariance), AND emitted as a structured table hanging off that clause (parent_index)."""
    fd = {
        "document_id": "doc",
        "units": [
            {"unit_id": "u1", "block_type": "text_block", "parent_unit_id": None,
             "source_block_ids": ["doc:p0001:b0001"],
             "content": {"text": "3. Duty rates are in Schedule-2."}},
            {"unit_id": "u2", "block_type": "table", "parent_unit_id": "u1",
             "source_block_ids": ["doc:p0001:b0002"],
             "content": {"table": {"type": "generic_table",
                                   "headers": ["Item", "Rate"],
                                   "rows": [["Rice", "5%"], ["Salt", "0%"]]}}},
        ],
    }
    pred = flatten_final_document(fd)

    # contents unchanged: one clause item whose text still inlines the table cells.
    assert len(pred["contents"]) == 1
    item = pred["contents"][0]
    assert item["type"] == "clause"
    assert "Duty rates" in item["text"] and "Rice" in item["text"] and "5%" in item["text"]

    # tables[]: one structured entry hanging off the clause (parent_index -> contents[0]).
    assert len(pred["tables"]) == 1
    tbl = pred["tables"][0]
    assert tbl["headers"] == ["Item", "Rate"]
    assert tbl["rows"] == [["Rice", "5%"], ["Salt", "0%"]]
    assert tbl["parent_index"] == 0


def test_standalone_table_is_metadata_and_structured():
    fd = {
        "document_id": "doc",
        "units": [
            {"unit_id": "t1", "block_type": "table", "parent_unit_id": None,
             "source_block_ids": ["doc:p0001:b0001"],
             "content": {"table": {"headers": ["A"], "rows": [["1"]]}}},
        ],
    }
    pred = flatten_final_document(fd)
    assert pred["contents"][0]["type"] == "metadata"          # a standalone table is furniture
    assert pred["tables"][0]["headers"] == ["A"]
    assert pred["tables"][0]["parent_index"] == 0             # points at its own inlined item


def test_fallback_markdown_html_table_slot():
    fd = {
        "document_id": "doc",
        "units": [
            {"unit_id": "c1", "block_type": "text_block", "parent_unit_id": None,
             "source_block_ids": ["b1"], "content": {"text": "Clause."}},
            {"unit_id": "t1", "block_type": "table", "parent_unit_id": "c1",
             "source_block_ids": ["b2"],
             "content": {"markdown": "| a | b |",
                         "html": "<table><tr><td>a</td></tr></table>",
                         "parsing_status": "fallback"}},
        ],
    }
    pred = flatten_final_document(fd)
    tbl = pred["tables"][0]
    assert "rows" not in tbl                                  # unstructured fallback
    assert tbl["markdown"] == "| a | b |"
    assert tbl["html"] == "<table><tr><td>a</td></tr></table>"
    assert tbl["parent_index"] == 0


def test_cross_page_stitched_table_merges_fragments():
    """A stitched (cross-page) table unit: content is a list of per-page fragments; they merge
    into ONE structured table (headers from the first, rows concatenated)."""
    fd = {
        "document_id": "doc",
        "units": [
            {"unit_id": "s1", "block_type": "table", "parent_unit_id": None,
             "source_block_ids": ["doc:p0001:b0001", "doc:p0002:b0001"],
             "content": [
                 {"block_type": "table",
                  "content": {"table": {"headers": ["N", "Item"], "rows": [["1", "x"]]}}},
                 {"block_type": "table",
                  "content": {"table": {"headers": ["N", "Item"], "rows": [["2", "y"]]}}},
             ]},
        ],
    }
    pred = flatten_final_document(fd)
    assert len(pred["tables"]) == 1
    tbl = pred["tables"][0]
    assert tbl["headers"] == ["N", "Item"]
    assert tbl["rows"] == [["1", "x"], ["2", "y"]]


def test_no_tables_yields_empty_list():
    fd = {
        "document_id": "doc",
        "units": [
            {"unit_id": "u1", "block_type": "text_block", "parent_unit_id": None,
             "source_block_ids": ["b1"], "content": {"text": "Just a clause."}},
        ],
    }
    pred = flatten_final_document(fd)
    assert pred["tables"] == []
    assert pred["contents"][0]["text"] == "Just a clause."

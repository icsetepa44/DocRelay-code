"""Tests for the eval bridge (src/eval/adapt.py) and the shared table helpers.

Invariants: clause `text` is PROSE ONLY (FIX-1b — tables are scored by the table metrics), `full_text`
adds the inline table text (doc-level metrics), and every table is ALSO a structured `tables[]` entry
hanging off its parent clause (T3)."""

from types import SimpleNamespace

from src.eval.adapt import adapt_final_document, block_page, route_fragment, _table_entry
from src.stages._common import block_text, table_struct, table_to_text


def _unit(uid, bt, content, parent=None, sources=()):
    return {"unit_id": uid, "block_type": bt, "content": content, "parent_unit_id": parent,
            "source_block_ids": list(sources)}


# --- shared helpers -------------------------------------------------------------------

def test_table_to_text_flattens_headers_and_rows():
    content = {"table": {"headers": ["H1", "H2"], "rows": [["a", "b"], ["c", "d"]]}}
    assert table_to_text(content) == "H1 H2 a b c d"


def test_table_to_text_preserves_empty_cells():
    """Empty cells are joined verbatim (extra space), NOT filtered out: a form row like `(ক) | `
    (empty 2nd column) must render `(ক) `, not collapse the gap."""
    content = {"table": {"headers": ["Name", "Date"], "rows": [["(a)", ""], ["(b)", ""]]}}
    assert table_to_text(content) == "Name Date (a)  (b)"


def test_block_text_prefers_text_then_falls_back_to_table():
    """block_text uses text/description when present, else flattens table cells (T6)."""
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
    assert table_struct({"table": {"headers": ["H"], "rows": [["x"]]}}) == {"headers": ["H"], "rows": [["x"]]}
    assert table_struct({"markdown": "x"}) is None
    assert table_struct({}) is None


# --- fragment routing (FIX-1b / FIX-3) ------------------------------------------------

def test_route_fragment_shapes():
    assert route_fragment("text_block", {"markdown": "(২১) প্রজ্ঞাপন"})["prose"] == "(২১) প্রজ্ঞাপন"
    assert route_fragment("text_block", {"html": "<p>hi</p>"})["prose"] == "<p>hi</p>"
    r = route_fragment("text_block", {"table": {"headers": ["a"], "rows": [["b", ""]]}, "parsing_status": "ok"})
    assert r["prose"] == "" and r["table"] == "a b" and r["recovered"]
    assert route_fragment("table", {"text": "cell text"})["table"] == "cell text"
    html = "<table><tr><td>2924.19.00</td><td>Carbromal</td></tr></table>"
    assert route_fragment("table", {"table": html})["table"] == html  # a table stored as a string
    assert _table_entry([{"table": html}]) == {"html": html}
    assert route_fragment("heading", {"text": "H"}) == {"prose": "H", "table": "", "recovered": False}


# --- adapt_final_document -------------------------------------------------------------

def test_child_table_is_prose_free_and_structured():
    """A table child of a clause: kept OUT of the clause's prose `text`, present in `full_text`, and
    emitted as a structured table hanging off that clause (parent_index)."""
    fd = {"units": [
        _unit("u1", "text_block", {"text": "3. Duty rates are in Schedule-2."}, sources=["b1"]),
        _unit("u2", "table", {"table": {"type": "generic_table", "headers": ["Item", "Rate"],
                                        "rows": [["Rice", "5%"], ["Salt", "0%"]]}}, parent="u1", sources=["b2"]),
    ]}
    pred = adapt_final_document(fd)
    assert len(pred["contents"]) == 1
    item = pred["contents"][0]
    assert item["type"] == "clause" and item["text"] == "3. Duty rates are in Schedule-2."
    assert "Rice" in item["full_text"] and "5%" in item["full_text"]
    assert pred["tables"] == [{"headers": ["Item", "Rate"], "rows": [["Rice", "5%"], ["Salt", "0%"]],
                               "parent_index": 0}]


def test_prose_only_clause_markdown_recovery_and_table_items():
    fd = {"units": [
        _unit("u1", "text_block", {"text": "1. Exempt goods:"}),
        _unit("u2", "table", {"table": {"headers": ["HS"], "rows": [["01.01"]]}}, parent="u1"),
        _unit("u3", "text_block", {"markdown": "2. Recovered markdown clause."}),
        _unit("u4", "text_block", {"table": {"headers": ["Q"], "rows": [["yes"]]}}),
        _unit("u5", "noise_or_separator", {"text": "-----"}),
    ]}
    out = adapt_final_document(fd)
    c = out["contents"]
    assert [x["type"] for x in c] == ["clause", "clause", "table"]
    assert c[0]["text"] == "1. Exempt goods:" and "01.01" in c[0]["full_text"]
    assert c[1]["text"] == "2. Recovered markdown clause."
    assert len(out["tables"]) == 2 and out["tables"][0]["parent_index"] == 0
    assert out["adapter_stats"].get("unrouted_fragments", 0) == 0
    assert out["adapter_stats"]["recovered_fragments"] == 2


def test_standalone_table_is_a_table_item_and_structured():
    pred = adapt_final_document({"units": [_unit("t1", "table", {"table": {"headers": ["A"], "rows": [["1"]]}})]})
    assert pred["contents"] == [{"type": "table", "text": "", "full_text": "A 1", "unit_id": "t1",
                                 "page_start": None, "page_end": None}]
    assert pred["tables"][0]["headers"] == ["A"] and pred["tables"][0]["parent_index"] == 0


def test_fallback_markdown_html_table_slot():
    fd = {"units": [
        _unit("c1", "text_block", {"text": "Clause."}),
        _unit("t1", "table", {"markdown": "| a | b |", "html": "<table><tr><td>a</td></tr></table>",
                              "parsing_status": "fallback"}, parent="c1"),
    ]}
    tbl = adapt_final_document(fd)["tables"][0]
    assert "rows" not in tbl                                  # unstructured fallback
    assert tbl["markdown"] == "| a | b |"
    assert tbl["html"] == "<table><tr><td>a</td></tr></table>"
    assert tbl["parent_index"] == 0


def test_cross_page_stitched_table_merges_fragments():
    """A stitched (cross-page) table unit: its per-page fragments merge into ONE structured table
    (headers from the first, rows concatenated)."""
    fd = {"units": [_unit("s1", "table", [
        {"block_type": "table", "content": {"table": {"headers": ["N", "Item"], "rows": [["1", "x"]]}}},
        {"block_type": "table", "content": {"table": {"headers": ["N", "Item"], "rows": [["2", "y"]]}}},
    ])]}
    pred = adapt_final_document(fd)
    assert len(pred["tables"]) == 1
    assert pred["tables"][0]["headers"] == ["N", "Item"]
    assert pred["tables"][0]["rows"] == [["1", "x"], ["2", "y"]]


def test_furniture_only_clause_is_demoted_to_metadata():
    """A3: a text_block whose source blocks are purely furniture skills is metadata; any other
    skill (or no attribution) keeps it a clause."""
    fd = {"units": [_unit("u1", "text_block", {"text": "Government seal"}, sources=["b1"]),
                    _unit("u2", "text_block", {"text": "1. A real clause."}, sources=["b2"])]}
    skills = {"b1": {"figure_or_seal"}, "b2": {"figure_or_seal", "numbered_point"}}
    assert [c["type"] for c in adapt_final_document(fd, skills)["contents"]] == ["metadata", "clause"]


def test_no_tables_yields_empty_list():
    pred = adapt_final_document({"units": [_unit("u1", "text_block", {"text": "Just a clause."})]})
    assert pred["tables"] == []
    assert pred["contents"][0]["text"] == "Just a clause."


# --- page provenance ------------------------------------------------------------------

def test_block_page_parses_zero_based_page():
    assert block_page("doc:p0003:b0012") == 3
    assert block_page("doc:s000001") is None
    assert block_page("b1") is None


def test_contents_carry_one_based_page_range_over_descendants():
    """page_start/page_end span every source block of the top-level unit AND its descendants,
    converted from the 0-based block-id page to the 1-based gold convention."""
    fd = {"units": [
        _unit("doc:p0000:b0001", "text_block", {"text": "1. Main."}, sources=["doc:p0000:b0001"]),
        _unit("doc:s000001", "text_block", [
            {"block_type": "text_block", "content": {"text": "(a) part one"}},
            {"block_type": "text_block", "content": {"text": "part two"}}],
            parent="doc:p0000:b0001", sources=["doc:p0000:b0002", "doc:p0002:b0001"]),
        _unit("doc:p0002:b0002", "text_block", {"text": "2. Next."}, sources=["doc:p0002:b0002"]),
    ]}
    c = adapt_final_document(fd)["contents"]
    assert [(x["unit_id"], x["page_start"], x["page_end"]) for x in c] == [
        ("doc:p0000:b0001", 1, 3), ("doc:p0002:b0002", 3, 3)]


# --- reading order --------------------------------------------------------------------

def test_stitched_unit_continuing_after_its_children_is_written_in_page_order():
    """A clause stitched onto the next page, whose sub-points sit on the first page, is written in page
    order: the page-2 fragment comes after the page-1 sub-points, not straight after the clause head."""
    fd = {"units": [
        _unit("doc:s000001", "text_block", [
            {"block_type": "text_block", "content": {"text": "25. (1) one"}},
            {"block_type": "text_block", "content": {"text": "(5) five"}}],
            sources=["doc:p0000:b0001", "doc:p0001:b0001"]),
        _unit("doc:p0000:b0002", "text_block", {"text": "(2) two"}, parent="doc:s000001", sources=["doc:p0000:b0002"]),
        _unit("doc:p0000:b0003", "text_block", {"text": "(3) three"}, parent="doc:s000001", sources=["doc:p0000:b0003"]),
        _unit("doc:p0000:b0004", "text_block", {"text": "(4) four"}, parent="doc:s000001", sources=["doc:p0000:b0004"]),
        _unit("doc:p0001:b0002", "text_block", {"text": "26. Next."}, sources=["doc:p0001:b0002"]),
    ]}
    c = adapt_final_document(fd)["contents"]
    assert c[0]["text"] == "25. (1) one\n(2) two\n(3) three\n(4) four\n(5) five"
    assert c[0]["full_text"] == c[0]["text"]
    assert [(x["type"], x["page_start"], x["page_end"]) for x in c] == [("clause", 1, 2), ("clause", 2, 2)]

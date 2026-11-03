import json

import pytest

from src.core import config
from src.eval import mpdocbench as mpdb

pytestmark = pytest.mark.skipif(not (config.MPDOCBENCH_REPO / "metrics").is_dir(),
                                reason="MPDocBench repo not found (data/mpdocbench/repo/MPDocBench)")
from src.eval.mpdocbench import GoldDoc, PredDoc, PredElement
from src.stages._common import table_to_text


def _fd():
    return {"units": [
        {"unit_id": "d:p0001:b0001", "block_type": "heading", "parent_unit_id": None,
         "source_block_ids": ["d:p0001:b0001"], "content": {"text": "Intro", "level": 1}},
        {"unit_id": "d:p0001:b0002", "block_type": "text_block", "parent_unit_id": None,
         "source_block_ids": ["d:p0001:b0002"], "content": {"text": "Hello world."}},
        {"unit_id": "d:p0001:b0003", "block_type": "table", "parent_unit_id": None,
         "source_block_ids": ["d:p0001:b0003"],
         "content": {"table": {"headers": ["A", "B"], "rows": [["1", "2"]]},
                     "parsing_status": "success"}},
        {"unit_id": "d:p0001:b0004", "block_type": "equation", "parent_unit_id": None,
         "source_block_ids": ["d:p0001:b0004"], "content": {"text": "$$E=mc^2$$"}},
    ]}


_TABLE_HTML = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"


def _gold(with_truncation=False):
    dets = [
        {"anno_id": 1, "category_type": "title", "page_id": 1, "order": 1,
         "poly": [0, 0, 10, 0, 10, 10, 0, 10], "text": "Intro"},
        {"anno_id": 2, "category_type": "text_block", "page_id": 1, "order": 2, "text": "Hello world."},
        {"anno_id": 3, "category_type": "table", "page_id": 1, "order": 3, "html": _TABLE_HTML},
        {"anno_id": 4, "category_type": "equation_isolated", "page_id": 1, "order": 4,
         "latex": "$$E=mc^2$$"},
        {"anno_id": 5, "category_type": "figure", "page_id": 1, "order": 5,
         "poly": [100, 100, 300, 100, 300, 300, 100, 300]},
    ]
    relations = [{"source_anno_id": "root", "target_anno_id": 1, "relation_type": "parent_son"}]
    if with_truncation:
        dets += [
            {"anno_id": 10, "category_type": "text_block", "page_id": 1, "order": 6, "text": "part one "},
            {"anno_id": 11, "category_type": "text_block", "page_id": 2, "order": 7, "text": "part two"},
        ]
        relations.append({"source_anno_id": 10, "target_anno_id": 11, "relation_type": "truncated"})
    return {
        "page_info": {"image_path": "testdoc.pdf", "width": [1000, 1000], "height": [1000, 1000],
                      "page_attribute": {"data_source": "book", "language": "english"}},
        "layout_dets": dets,
        "extra": {"relation": relations, "pdf_name": "hash"},
    }


def test_leaf_scorers_import_and_behave():
    L = mpdb._load_leaf()
    for k in ("TEDS", "HeadTEDS", "clean_string", "textblock2unicode",
              "normalized_table", "merge_tables", "relation_utils", "match_gt2pred", "lev_distance"):
        assert k in L, f"leaf handle missing: {k}"
    norm = L["normalized_table"](_TABLE_HTML, format="html")
    assert "<body>" in norm
    assert L["TEDS"](structure_only=False).evaluate(norm, norm) == pytest.approx(1.0)
    score, _depth = L["HeadTEDS"]().evaluate(["# A", "## B"], ["# A", "## B"])
    assert score == pytest.approx(1.0)
    assert L["textblock2unicode"]("plain text") == "plain text"


def test_gold_loader_truncation_and_heads():
    g = mpdb.build_gold_doc(_gold(with_truncation=True))
    assert len(g.merged_text) == 1 and g.merged_text[0]["cross_pages"] == 2
    assert g.merged_text[0]["text"] == "part one part two"
    parent_son, head_nodes = g.head_input
    assert g.n_titles == 1 and set(head_nodes) == {1} and parent_son == [
        {"source_anno_id": "root", "target_anno_id": 1, "relation_type": "parent_son"}]
    assert len(g.tables) == 1
    assert any(t["category"] == "title" for t in g.text_elems)


def test_pred_builder_maps_categories_and_shapes():
    pred = mpdb.build_pred_doc(_fd(), "testdoc")
    cats = [e.category for e in pred.elements]
    assert cats == ["title", "text_block", "table", "equation_isolated"]
    assert pred.head_md == ["# Intro"]
    tbl = next(e for e in pred.elements if e.block_type == "table")
    assert "<td>A</td>" in tbl.html and "<td>1</td>" in tbl.html
    assert pred.elements[0].page == 1 and pred.elements[0].seq == 1


def test_grid_to_html_and_pred_table_html_fallback():
    assert mpdb._grid_to_html([["A", "B"], ["1", "2"]]) == _TABLE_HTML
    unit = {"block_type": "table", "content": {"html": "<table><tr><td>x</td></tr></table>",
                                               "parsing_status": "fallback"}}
    assert mpdb._pred_table_html(unit) == "<table><tr><td>x</td></tr></table>"


def test_pred_markdown_export():
    md = mpdb.pred_markdown(_fd())
    assert md == "# Intro\n\nHello world.\n\n" + _TABLE_HTML + "\n\n$$E=mc^2$$"


def test_text_edit_identical_and_perturbed():
    same = mpdb.text_edit.score(_gold(), mpdb.pred_markdown(_fd()), "testdoc.pdf")
    assert same["edit_dist"] == pytest.approx(0.0) and same["score"] == pytest.approx(1.0)
    assert same["n_blocks"] >= 2 and set(same) == {"edit_num", "upper", "edit_dist", "score", "n_blocks"}
    fd2 = _fd()
    fd2["units"] = [u for u in fd2["units"] if u["block_type"] != "text_block"]
    assert mpdb.text_edit.score(_gold(), mpdb.pred_markdown(fd2), "testdoc.pdf")["score"] < 1.0


def test_text_diag_flags_over_production():
    g = mpdb.build_gold_doc(_gold())
    fd = _fd()
    fd["units"].append({"unit_id": "d:p0001:b0009", "block_type": "text_block", "parent_unit_id": None,
                        "source_block_ids": ["d:p0001:b0009"],
                        "content": {"text": "a wholly spurious hallucinated extra paragraph"}})
    withextra = mpdb.match_text(mpdb.build_pred_doc(fd, "testdoc"), g)
    assert mpdb.score_text_diag(withextra)["precision"] < 1.0


def test_score_tables_and_heads_identical():
    g = mpdb.build_gold_doc(_gold())
    p = mpdb.build_pred_doc(_fd(), "testdoc")
    t = mpdb.score_tables(p, g, mpdb.match_tables(p, g))
    assert t["teds"] == pytest.approx(1.0) and t["n_gold"] == 1 and t["n_pred"] == 1
    h = mpdb.score_heads(p, g)
    assert h["score"] == pytest.approx(1.0) and h["n_gold_titles"] == 1


def _ro_gold():
    texts = ["alpha first block", "beta second block", "gamma third block", "delta fourth block"]
    return texts, GoldDoc(text_elems=[{"page_id": 1, "order": i + 1, "category": "text_block", "text": t}
                                      for i, t in enumerate(texts)])


def _ro_pred(texts, order):
    return PredDoc(elements=[PredElement(unit_id=str(i), block_type="text_block", category="text_block",
                                         page=1, seq=i, text=texts[j]) for i, j in enumerate(order)])


def test_score_reading_order():
    texts, gold = _ro_gold()
    s_ident = mpdb.score_reading_order(mpdb.match_text_document(_ro_pred(texts, [0, 1, 2, 3]), gold))["score"]
    s_rev = mpdb.score_reading_order(mpdb.match_text_document(_ro_pred(texts, [3, 2, 1, 0]), gold))["score"]
    s_miss = mpdb.score_reading_order(mpdb.match_text_document(_ro_pred(texts, [0, 1, 2]), gold))["score"]
    assert s_ident == 1.0
    assert s_ident > s_rev
    assert s_ident > s_miss
    assert 0.0 <= s_rev <= s_ident <= 1.0


def test_reading_order_merged_block_not_penalised():
    texts, gold = _ro_gold()
    merged = PredDoc(elements=[
        PredElement(unit_id="m", block_type="text_block", category="text_block", page=1, seq=0,
                    text=texts[0] + " " + texts[1]),
        PredElement(unit_id="c", block_type="text_block", category="text_block", page=1, seq=1, text=texts[2]),
        PredElement(unit_id="d", block_type="text_block", category="text_block", page=1, seq=2, text=texts[3])])
    ro = mpdb.score_reading_order(mpdb.match_text_document(merged, gold))
    assert ro["score"] == 1.0 and ro["n_gold"] == 4


def test_reading_order_matches_across_pages():
    texts = ["alpha first block", "beta second block"]
    gold = GoldDoc(text_elems=[{"page_id": p, "order": p, "category": "text_block", "text": t}
                               for p, t in ((1, texts[0]), (2, texts[1]))])
    pred = PredDoc(elements=[PredElement(unit_id=str(p), block_type="text_block", category="text_block",
                                         page=p, seq=0, text=t) for p, t in ((1, texts[1]), (2, texts[0]))])
    ro = mpdb.score_reading_order(mpdb.match_text_document(pred, gold))
    assert ro["n_matched"] == 2 and ro["score"] < 1.0


def test_reading_order_is_mean_of_documents():
    per_doc = [{"doc": "a", "reading_order": {"edit_num": 30, "upper": 100, "score": 0.7}},
               {"doc": "b", "reading_order": {"edit_num": 0, "upper": 10, "score": 1.0}}]
    assert mpdb._aggregate(per_doc, {})["overall"]["reading_order"] == pytest.approx(0.85)


def test_score_text_diag_under_and_over_segmentation():
    texts, gold = _ro_gold()
    under = PredDoc(elements=[
        PredElement(unit_id="m", block_type="text_block", category="text_block", page=1, seq=0,
                    text=texts[0] + " " + texts[1]),
        PredElement(unit_id="c", block_type="text_block", category="text_block", page=1, seq=1,
                    text=texts[2])])
    du = mpdb.score_text_diag(mpdb.match_text(under, gold))
    assert du["n_gold"] == 4 and du["recall"] < 1.0
    over = _ro_pred(texts, [0, 1, 2, 3])
    over.elements.append(PredElement(unit_id="x", block_type="text_block", category="text_block",
                                     page=1, seq=9, text="spurious extra paragraph text"))
    do = mpdb.score_text_diag(mpdb.match_text(over, gold))
    assert do["n_pred"] == 5 and do["precision"] < 1.0 and do["recall"] == pytest.approx(1.0)


def test_leading_text_empty_page_does_not_shift_alignment():
    gold = GoldDoc(
        text_elems=[{"page_id": 1, "order": 1, "category": "text_block", "text": "alpha first block"},
                    {"page_id": 2, "order": 2, "category": "text_block", "text": "beta second block"}],
        page_universe=[1, 2])
    pred = PredDoc(
        elements=[PredElement(unit_id="u", block_type="text_block", category="text_block",
                              page=2, seq=1, text="beta second block")],
        page_universe=[1, 2])
    diag = mpdb.score_text_diag(mpdb.match_text(pred, gold))
    assert diag["n_gold"] == 2 and diag["n_pred"] == 1
    assert diag["n_true_matches"] == 1


def test_page_present_on_one_side_only_does_not_shift_aligned_pages():
    gold = GoldDoc(
        text_elems=[{"page_id": 1, "order": 1, "category": "text_block", "text": "alpha block one"},
                    {"page_id": 3, "order": 1, "category": "text_block", "text": "gamma block three"}],
        page_universe=[1, 3])
    pred = PredDoc(
        elements=[PredElement(unit_id="a", block_type="text_block", category="text_block",
                              page=0, seq=1, text="alpha block one"),
                  PredElement(unit_id="c", block_type="text_block", category="text_block",
                              page=2, seq=1, text="gamma block three")],
        page_universe=[0, 1, 2])
    diag = mpdb.score_text_diag(mpdb.match_text(pred, gold))
    assert diag["n_true_matches"] == 2


def test_builders_populate_page_universe():
    assert mpdb.build_gold_doc(_gold(with_truncation=True)).page_universe == [1, 2]
    assert mpdb.build_gold_doc(_gold()).page_universe == [1]
    assert mpdb.build_pred_doc(_fd(), "testdoc").page_universe == [1]


def test_score_merged_text_and_table():
    g = mpdb.build_gold_doc(_gold(with_truncation=True))
    p = PredDoc(elements=[PredElement(unit_id="u", block_type="text_block", category="text_block",
                                      page=1, seq=1, text="part one part two", merged=True)])
    assert mpdb.score_merged_text(p, g)["score"] == pytest.approx(1.0)
    assert mpdb.score_merged_text(p, mpdb.build_gold_doc(_gold())) is None
    gt = GoldDoc(merged_tables=[{"order": 1, "html": _TABLE_HTML, "is_merged": "table"}])
    pt = PredDoc(elements=[PredElement(unit_id="t", block_type="table", category="table",
                                       page=1, seq=1, html=_TABLE_HTML, merged=True)])
    assert mpdb.score_merged_table(pt, gt)["teds"] == pytest.approx(1.0)


def test_merged_text_credits_within_page_inline_merge():
    g = mpdb.build_gold_doc(_gold(with_truncation=True))
    inline = PredDoc(elements=[PredElement(unit_id="u", block_type="text_block",
                                           category="text_block", page=1, seq=1,
                                           text="part one part two", merged=False)])
    res = mpdb.score_merged_text(inline, g)
    assert res["score"] == pytest.approx(1.0)
    assert res["n_reconstructed"] == 1
    assert res["n_pred_merges"] == 0
    split = PredDoc(elements=[
        PredElement(unit_id="a", block_type="text_block", category="text_block", page=1, seq=1,
                    text="part one", merged=False),
        PredElement(unit_id="b", block_type="text_block", category="text_block", page=1, seq=2,
                    text="part two", merged=False)])
    assert mpdb.score_merged_text(split, g)["score"] < 0.8


def test_merged_table_credits_single_page_table():
    gt = GoldDoc(merged_tables=[{"order": 1, "html": _TABLE_HTML, "is_merged": "table"}])
    pt = PredDoc(elements=[PredElement(unit_id="t", block_type="table", category="table",
                                       page=1, seq=1, html=_TABLE_HTML, merged=False)])
    res = mpdb.score_merged_table(pt, gt)
    assert res["teds"] == pytest.approx(1.0)
    assert res["n_reconstructed"] == 1 and res["n_pred_merges"] == 0


def test_score_relation_optional():
    g = _gold()
    g["layout_dets"] += [
        {"anno_id": 20, "category_type": "table", "page_id": 1, "order": 8,
         "html": "<table><tr><td>r1</td></tr></table>"},
        {"anno_id": 21, "category_type": "table", "page_id": 2, "order": 9,
         "html": "<table><tr><td>r2</td></tr></table>"},
    ]
    g["extra"]["relation"].append({"source_anno_id": 20, "target_anno_id": 21,
                                   "relation_type": "truncated"})
    p = PredDoc(elements=[PredElement(unit_id="m", block_type="table", category="table", page=1,
                                      seq=1, html="<table><tr><td>r1</td><td>r2</td></tr></table>",
                                      merged=True)])
    res = mpdb.score_relation(p, g)
    assert res is None or ("f1" in res and 0.0 <= res["f1"] <= 1.0)


def test_evaluate_doc_end_to_end():
    res = mpdb.evaluate_doc("testdoc", _fd(), _gold(with_truncation=True))
    assert res["doc"] == "testdoc"
    for key in ("text", "table", "heads", "reading_order", "merged_text", "merged_table", "counts"):
        assert key in res
    for _name, path in mpdb._PRIMARY:
        v = mpdb._dig(res, path)
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{path} out of range: {v}"


def test_evaluate_run_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(mpdb, "load_gold_index", lambda: {})
    monkeypatch.setattr(mpdb, "_domain_map", lambda stems: {})
    res = mpdb.evaluate_run(tmp_path)
    assert res["models"] == {}
    assert (tmp_path / "_mpdocbench" / "metrics.json").is_file()


def test_evaluate_run_scores_synthetic_doc(tmp_path, monkeypatch):
    fd_dir = tmp_path / "testdoc" / "gemma"
    fd_dir.mkdir(parents=True)
    (fd_dir / "final_document.json").write_text(json.dumps(_fd()), encoding="utf-8")
    monkeypatch.setattr(mpdb, "load_gold_index", lambda: {"testdoc": _gold()})
    monkeypatch.setattr(mpdb, "_domain_map", lambda stems: {"testdoc": "book"})
    res = mpdb.evaluate_run(tmp_path)
    assert "gemma" in res["models"]
    agg = res["models"]["gemma"]["aggregate"]
    assert agg["n_docs"] == 1 and agg["overall"]["text"] == pytest.approx(1.0)
    assert "book" in agg["by_domain"]


_TABLE_HTML_2 = "<table><tr><td>C</td><td>D</td></tr><tr><td>3</td><td>4</td></tr></table>"
_UNRELATED_TABLE = "<table><tr><td>a wholly unrelated paragraph of prose text here</td></tr></table>"


def _gold_without_tables():
    g = _gold()
    g["layout_dets"] = [d for d in g["layout_dets"] if d["category_type"] != "table"]
    return g


def _tbl_elem(uid, html, seq, text=""):
    return PredElement(unit_id=uid, block_type="table", category="table", page=1, seq=seq,
                       html=html, table_text=text)


def test_build_pred_doc_captures_table_text():
    tbl = next(e for e in mpdb.build_pred_doc(_fd(), "testdoc").elements if e.block_type == "table")
    assert tbl.table_text == table_to_text({"table": {"headers": ["A", "B"], "rows": [["1", "2"]]}})
    assert tbl.table_text


def test_match_tables_is_order_independent():
    g = GoldDoc(tables=[{"html": _TABLE_HTML}, {"html": _TABLE_HTML_2}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML_2, 1), _tbl_elem("p1", _TABLE_HTML, 2)])
    m = mpdb.match_tables(p, g)
    r = mpdb.score_tables(p, g, m)
    assert r["teds"] == pytest.approx(1.0) and r["n_matched"] == 2 and not m.unmatched_pred


def test_match_tables_missed_gold_scores_zero():
    g = GoldDoc(tables=[{"html": _TABLE_HTML}, {"html": _TABLE_HTML_2}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML, 1)])
    m = mpdb.match_tables(p, g)
    r = mpdb.score_tables(p, g, m)
    assert r["n_gold"] == 2 and r["n_matched"] == 1
    assert r["teds"] == pytest.approx(0.5, abs=0.05)


def test_apply_redirect_when_gold_has_no_tables():
    gold = mpdb.build_gold_doc(_gold_without_tables())
    assert gold.tables == []
    pred = mpdb.build_pred_doc(_fd(), "testdoc")
    before = len(mpdb._pred_text_entries(pred))
    m = mpdb.match_tables(pred, gold)
    assert len(m.unmatched_pred) == 1 and m.n_gold == 0
    mpdb.apply_table_text_redirect(pred, m)
    assert not any(e.block_type == "table" for e in pred.elements)
    demoted = next(e for e in pred.elements if "1" in e.text and "2" in e.text)
    assert demoted.html == "" and demoted.block_type == "text_block" and demoted.category == "text_block"
    assert len(mpdb._pred_text_entries(pred)) == before + 1


def test_redirect_keeps_matched_table_and_redirects_extra():
    g = GoldDoc(tables=[{"html": _TABLE_HTML}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML, 1, "A B 1 2"),
                          _tbl_elem("p1", "<table><tr><td>Introduction ..... 3</td></tr></table>", 2,
                                    "Introduction ..... 3")])
    m = mpdb.match_tables(p, g)
    mpdb.apply_table_text_redirect(p, m)
    kept = [e for e in p.elements if e.block_type == "table"]
    assert len(kept) == 1 and "<td>A</td>" in kept[0].html
    assert any("Introduction" in e.text for e in p.elements if e.block_type == "text_block")
    assert mpdb.score_tables(p, g, m)["teds"] == pytest.approx(1.0)


def test_table_diag_flags_overprediction():
    g = GoldDoc(tables=[{"html": _TABLE_HTML}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML, 1),
                          _tbl_elem("p1", _UNRELATED_TABLE, 2)])
    d = mpdb.score_table_diag(mpdb.match_tables(p, g))
    assert d["n_pred"] == 2 and d["n_true_matches"] == 1
    assert d["precision"] == pytest.approx(0.5) and d["recall"] == pytest.approx(1.0)


def test_table_diag_flags_equalcount_misclassification():
    g = GoldDoc(tables=[{"html": _TABLE_HTML}])
    p = PredDoc(elements=[_tbl_elem("p0", _UNRELATED_TABLE, 1)])
    m = mpdb.match_tables(p, g)
    assert m.n_pred == 1 and len(m.matched_pred_idxs) == 1
    assert mpdb.score_tables(p, g, m)["teds"] < 0.5
    d = mpdb.score_table_diag(m)
    assert d["n_true_matches"] == 0 and d["f1"] == 0.0


def test_table_diag_none_when_no_tables_either_side():
    assert mpdb.score_table_diag(mpdb.match_tables(PredDoc(), GoldDoc())) is None


def test_evaluate_doc_redirects_unmatched_table_into_text():
    res = mpdb.evaluate_doc("testdoc", _fd(), _gold_without_tables())
    assert res["table"] is None
    assert res["text"]["n_blocks"] > 0 and res["text"]["score"] > 0.0
    assert res["text_diag"]["n_pred"] == 3
    assert res["table_diag"]["precision"] == 0.0 and res["table_diag"]["n_pred"] == 1

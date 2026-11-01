"""MPDocBench eval adapter. Hermetic — no LLM, no scored run, no 35MB gold load
(gold entries are hand-built; the real index is monkeypatched where needed). Needs the official MPDocBench
code (data/mpdocbench/repo/MPDocBench/, see data/README.md): its metric functions are imported, never copied.

Proves: leaf scorers import + behave (the bs4-style "dep not installed" gap), gold truncation/head/
figure loading, pred construction from a FinalDocument, every metric fn (identical -> perfect,
perturbed -> degraded), evaluate_doc end-to-end, and the weak-spot report / run wiring."""

import json
from pathlib import Path

import pytest

from src.core import config
from src.eval import mpdocbench as mpdb
from src.eval import cdm_formula

pytestmark = pytest.mark.skipif(not (config.MPDOCBENCH_REPO / "metrics").is_dir(),
                                reason="official MPDocBench code not downloaded (data/mpdocbench/repo/MPDocBench)")
from src.eval.mpdocbench import GoldDoc, PredDoc, PredElement
from src.stages._common import table_to_text


# --------------------------------------------------------------------------- fixtures
def _fd():
    """A tiny FinalDocument dict: heading, paragraph, structured table, equation (one page)."""
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
        # two text fragments on pages 1 and 2 that form one cross-page paragraph
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


# --------------------------------------------------------------------------- 1. deps + shim
def test_leaf_scorers_import_and_behave():
    L = mpdb._load_leaf()
    for k in ("TEDS", "HeadTEDS", "clean_string", "textblock2unicode", "normalized_formula",
              "normalized_table", "merge_tables", "relation_utils", "match_gt2pred", "lev_distance"):
        assert k in L, f"leaf handle missing: {k}"
    # TEDS on an identical table == 1.0 (via the real path: normalize -> evaluate; TEDS needs the
    # <html><body><table> wrapper that normalized_table adds).
    norm = L["normalized_table"](_TABLE_HTML, format="html")
    assert "<body>" in norm  # normalizer wraps the fragment
    assert L["TEDS"](structure_only=False).evaluate(norm, norm) == pytest.approx(1.0)
    # HeadTEDS on identical markdown trees == 1.0
    score, _depth = L["HeadTEDS"]().evaluate(["# A", "## B"], ["# A", "## B"])
    assert score == pytest.approx(1.0)
    # normalized_formula strips $ delimiters and \left/\right (pure string ops, no latexmlc)
    out = L["normalized_formula"](r"$$\left(x\right)$$")
    assert "$" not in out and "left" not in out and "right" not in out
    assert L["textblock2unicode"]("plain text") == "plain text"


# --------------------------------------------------------------------------- 2. gold loader
def test_gold_loader_truncation_heads_figures():
    g = mpdb.build_gold_doc(_gold(with_truncation=True))
    # truncation merge -> one merged text element tagged cross_pages=2
    assert len(g.merged_text) == 1 and g.merged_text[0]["cross_pages"] == 2
    assert g.merged_text[0]["text"] == "part one part two"
    # head tree pair: [parent_son_list, head_nodes]; only the title node is a head
    parent_son, head_nodes = g.head_input
    assert g.n_titles == 1 and set(head_nodes) == {1} and parent_son == [
        {"source_anno_id": "root", "target_anno_id": 1, "relation_type": "parent_son"}]
    # figure poly[0,1,4,5]=(100,100,300,300) / (w,h)=1000 -> 0-1000 normalized == [100,100,300,300]
    assert len(g.figures) == 1
    assert g.figures[0]["bbox_norm"] == pytest.approx([100.0, 100.0, 300.0, 300.0])
    assert g.figures[0]["page_rank"] == 0
    # categorized content
    assert len(g.tables) == 1 and len(g.equations) == 1
    assert any(t["category"] == "title" for t in g.text_elems)


# --------------------------------------------------------------------------- 3. pred builder
def test_pred_builder_maps_categories_and_shapes():
    pred = mpdb.build_pred_doc(_fd(), "testdoc")  # no layout dir -> figures == []
    cats = [e.category for e in pred.elements]
    assert cats == ["title", "text_block", "table", "equation_isolated"]
    assert pred.head_md == ["# Intro"]
    tbl = next(e for e in pred.elements if e.block_type == "table")
    assert "<td>A</td>" in tbl.html and "<td>1</td>" in tbl.html
    eq = next(e for e in pred.elements if e.block_type == "equation")
    assert eq.latex == "$$E=mc^2$$"
    assert pred.elements[0].page == 1 and pred.elements[0].seq == 1
    assert pred.figures == []


def test_grid_to_html_and_pred_table_html_fallback():
    assert mpdb._grid_to_html([["A", "B"], ["1", "2"]]) == _TABLE_HTML
    # html fallback content wins verbatim
    unit = {"block_type": "table", "content": {"html": "<table><tr><td>x</td></tr></table>",
                                               "parsing_status": "fallback"}}
    assert mpdb._pred_table_html(unit) == "<table><tr><td>x</td></tr></table>"


# --------------------------------------------------------------------------- 4. metric fns
def test_score_text_identical_and_perturbed():
    g = mpdb.build_gold_doc(_gold())
    p = mpdb.build_pred_doc(_fd(), "testdoc")
    assert mpdb.score_text(p, g)["score"] == pytest.approx(1.0)
    # drop the paragraph -> pred text shorter -> whole-string score < 1
    fd2 = _fd()
    fd2["units"] = [u for u in fd2["units"] if u["block_type"] != "text_block"]
    assert mpdb.score_text(mpdb.build_pred_doc(fd2, "testdoc"), g)["score"] < 1.0


def test_text_diag_flags_over_production():
    """The whole-string ``text`` metric is segmentation-blind, so a spurious extra pred paragraph is not
    isolated there; over-production shows up in the ``text_diag`` diagnostic instead (precision < 1)."""
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


def test_score_equations():
    g = mpdb.build_gold_doc(_gold())
    p = mpdb.build_pred_doc(_fd(), "testdoc")
    res = mpdb.score_equations(p, g)
    assert res["score"] == pytest.approx(1.0)
    assert res["method"] in ("cdm", "edit")
    # hermetic env (Windows conda, no toolchain) -> honest edit fallback, never mislabelled as CDM
    if not cdm_formula.cdm_available():
        assert res["method"] == "edit"
    # no gold equations -> None
    gold_no_eq = _gold()
    gold_no_eq["layout_dets"] = [d for d in gold_no_eq["layout_dets"]
                                 if d["category_type"] != "equation_isolated"]
    assert mpdb.score_equations(p, mpdb.build_gold_doc(gold_no_eq)) is None


def test_score_equations_diag():
    g = mpdb.build_gold_doc(_gold())
    p = mpdb.build_pred_doc(_fd(), "testdoc")
    d = mpdb.score_equations_diag(p, g)
    assert d["score"] == pytest.approx(1.0) and d["n_gold"] == 1 and d["n_pred"] == 1
    fd2 = _fd()
    fd2["units"][3]["content"]["text"] = "$$E=mc^3 + zzzzz$$"   # perturbed pred
    assert mpdb.score_equations_diag(mpdb.build_pred_doc(fd2, "testdoc"), g)["score"] < 1.0
    gold_no_eq = _gold()
    gold_no_eq["layout_dets"] = [x for x in gold_no_eq["layout_dets"]
                                 if x["category_type"] != "equation_isolated"]
    assert mpdb.score_equations_diag(p, mpdb.build_gold_doc(gold_no_eq)) is None


def test_equation_align_reuses_paper_matcher():
    """The adapter's pairing is the paper's real match_gt2pred_simple (pure Python, no toolchain)."""
    gold = [{"latex": "E=mc^2", "order": 1}, {"latex": "a^2+b^2=c^2", "order": 2}]
    pairs = cdm_formula.align(gold, ["E=mc^2", "x+y"], "d")
    assert ("E=mc^2", "E=mc^2") in pairs and len(pairs) == 2
    # missing pred -> gold paired with "" (CDM will score 0)
    assert cdm_formula.align([{"latex": "E=mc^2", "order": 1}], [], "d") == [("E=mc^2", "")]
    # extra/hallucinated pred dropped (only gold-anchored pairs survive)
    assert cdm_formula.align([{"latex": "E=mc^2", "order": 1}],
                             ["E=mc^2", "zzz", "qqq"], "d") == [("E=mc^2", "E=mc^2")]
    # paper's exact pre-CDM delimiter strip
    assert cdm_formula._strip_delims("$$ x^2 $$") == "x^2"
    assert cdm_formula._strip_delims("```latex\ny^2\n```", is_pred=True) == "y^2"
    assert cdm_formula._strip_delims(r"\[ z^2 \]") == "z^2"


def test_equation_fallback_when_cdm_unavailable(monkeypatch):
    g = mpdb.build_gold_doc(_gold())
    p = mpdb.build_pred_doc(_fd(), "testdoc")
    monkeypatch.setattr(mpdb.config, "EQUATION_METRIC", "cdm")
    monkeypatch.setattr(mpdb.cdm_formula, "cdm_available", lambda: False)
    res = mpdb.score_equations(p, g, "testdoc")
    assert res["method"] == "edit"
    assert res["score"] == pytest.approx(mpdb.score_equations_diag(p, g)["score"])


def test_equation_cdm_failure_degrades_to_edit(monkeypatch):
    """A wholesale CDM failure must degrade to the edit diagnostic, never raise."""
    g = mpdb.build_gold_doc(_gold())
    p = mpdb.build_pred_doc(_fd(), "testdoc")
    monkeypatch.setattr(mpdb.config, "EQUATION_METRIC", "cdm")
    monkeypatch.setattr(mpdb.cdm_formula, "cdm_available", lambda: True)

    def _boom(*a, **k):
        raise RuntimeError("render toolchain missing")
    monkeypatch.setattr(mpdb.cdm_formula, "score", _boom)
    res = mpdb.score_equations(p, g, "testdoc")
    assert res["method"] == "edit"


def test_cdm_work_dir_override(monkeypatch, tmp_path):
    """CDM scratch dir honors the space-free override; None keeps the default under the run dir.

    (Guards the fix for the vendored renderer's unquoted xelatex/ImageMagick paths breaking on a repo
    path that contains spaces.)"""
    run_dir = tmp_path / "run_x"
    monkeypatch.setattr(mpdb.config, "EQUATION_CDM_WORK_DIR", None)
    assert mpdb._resolve_cdm_work_dir(run_dir) == run_dir / "_mpdocbench" / "_cdm_work"
    monkeypatch.setattr(mpdb.config, "EQUATION_CDM_WORK_DIR", "/tmp/mpdocbench_cdm_work")
    assert mpdb._resolve_cdm_work_dir(run_dir) == Path("/tmp/mpdocbench_cdm_work")


def test_equation_micro_aggregation():
    """Overall equation = paper micro-mean (Sum sum_f1 / Sum n_pairs), not the doc-average."""
    docA = {"doc": "a", "equation": {"score": 1.0, "n_pairs": 1, "sum_f1": 1.0, "method": "cdm"}}
    docB = {"doc": "b", "equation": {"score": 0.0, "n_pairs": 3, "sum_f1": 0.0, "method": "cdm"}}
    agg = mpdb._aggregate([docA, docB], {"a": "paper", "b": "paper"})
    assert agg["overall"]["equation"] == pytest.approx(0.25)  # micro; macro would be 0.5


def test_equation_micro_none_when_edit_fallback():
    """An edit-fallback run keeps the macro mean (micro override returns None) and is labelled honestly."""
    docA = {"doc": "a", "equation": {"score": 0.5, "n_gold": 1, "n_pred": 1, "method": "edit"}}
    assert mpdb._equation_micro([docA]) is None
    agg = mpdb._aggregate([docA], {"a": "paper"})
    assert agg["overall"]["equation"] == pytest.approx(0.5)
    md = mpdb.render_weak_spots(
        {"run": "r", "skipped_no_gold": [], "models": {"m": {"per_doc": [docA], "aggregate": agg}}},
        {"a": "paper"})
    assert "edit-distance fallback" in md and "Equation(edit!)" in md


@pytest.mark.skipif(not cdm_formula.cdm_available(),
                    reason="CDM toolchain (xelatex/magick/node/skimage) absent — run under WSL")
def test_cdm_real(tmp_path):
    r = cdm_formula.score([{"latex": "x^2", "order": 1}], ["x^2"], "d", tmp_path)
    assert r["n_pairs"] == 1 and r["pair_f1"][0] == pytest.approx(1.0)


def test_score_figures_match_and_disjoint():
    gold = GoldDoc(figures=[{"page_id": 1, "page_rank": 0, "bbox_norm": [100, 100, 300, 300]}])
    hit = PredDoc(figures=[{"page_id": 1, "page_rank": 0, "bbox_norm": [100, 100, 300, 300]}])
    r = mpdb.score_figures(hit, gold)
    assert r["f1"] == pytest.approx(1.0) and r["matched"] == 1
    miss = PredDoc(figures=[{"page_id": 1, "page_rank": 0, "bbox_norm": [500, 500, 600, 600]}])
    assert mpdb.score_figures(miss, gold)["f1"] == 0.0
    # different page rank never matches
    wrong_page = PredDoc(figures=[{"page_id": 2, "page_rank": 1, "bbox_norm": [100, 100, 300, 300]}])
    assert mpdb.score_figures(wrong_page, gold)["matched"] == 0
    assert mpdb.score_figures(PredDoc(figures=[]), GoldDoc(figures=[])) is None


def _ro_gold():
    texts = ["alpha first block", "beta second block", "gamma third block", "delta fourth block"]
    return texts, GoldDoc(text_elems=[{"page_id": 1, "order": i + 1, "category": "text_block", "text": t}
                                      for i, t in enumerate(texts)])


def _ro_pred(texts, order):
    return PredDoc(elements=[PredElement(unit_id=str(i), block_type="text_block", category="text_block",
                                         page=1, seq=i, text=texts[j]) for i, j in enumerate(order)])


def test_score_reading_order():
    """Paper-faithful order edit over MATCHED text blocks: a genuine permutation of the *same* blocks
    degrades the score (the old category-symbol proxy was blind to reordering same-category blocks) and
    a missing block costs a deletion. Identical order scores exactly 1.0."""
    texts, gold = _ro_gold()
    s_ident = mpdb.score_reading_order(mpdb.match_text(_ro_pred(texts, [0, 1, 2, 3]), gold))["score"]
    s_rev = mpdb.score_reading_order(mpdb.match_text(_ro_pred(texts, [3, 2, 1, 0]), gold))["score"]
    s_miss = mpdb.score_reading_order(mpdb.match_text(_ro_pred(texts, [0, 1, 2]), gold))["score"]
    assert s_ident == 1.0    # perfect order, no index-0 deletion
    assert s_ident > s_rev   # a genuine permutation is now detected (proxy could not)
    assert s_ident > s_miss  # a missing block costs a deletion
    assert 0.0 <= s_rev <= s_ident <= 1.0


def test_reading_order_merged_block_not_penalised():
    """The official matcher (quick_match, configs/end2end.yaml) lets one pred block cover consecutive gold
    blocks, so a pred block that merges two gold blocks, in the right place, keeps a perfect order score.
    (The one-to-one simple matcher left the second gold block unmatched: a deletion.)"""
    texts, gold = _ro_gold()
    merged = PredDoc(elements=[
        PredElement(unit_id="m", block_type="text_block", category="text_block", page=1, seq=0,
                    text=texts[0] + " " + texts[1]),
        PredElement(unit_id="c", block_type="text_block", category="text_block", page=1, seq=1, text=texts[2]),
        PredElement(unit_id="d", block_type="text_block", category="text_block", page=1, seq=2, text=texts[3])])
    ro = mpdb.score_reading_order(mpdb.match_text(merged, gold))
    assert ro["score"] == 1.0 and ro["n_gold"] == 4


def test_score_text_diag_under_and_over_segmentation():
    """text_diag is the real segmentation signal the headline hides. Under-segmentation (fewer, merged
    pred blocks) lowers recall; over-production (extra pred blocks) lowers precision."""
    texts, gold = _ro_gold()  # 4 gold text blocks
    # under-segmentation: one pred block carrying two gold paragraphs' text -> few blocks match
    under = PredDoc(elements=[
        PredElement(unit_id="m", block_type="text_block", category="text_block", page=1, seq=0,
                    text=texts[0] + " " + texts[1]),
        PredElement(unit_id="c", block_type="text_block", category="text_block", page=1, seq=1,
                    text=texts[2])])
    du = mpdb.score_text_diag(mpdb.match_text(under, gold))
    assert du["n_gold"] == 4 and du["recall"] < 1.0
    # over-production: the 4 gold blocks plus a spurious extra -> precision < 1
    over = _ro_pred(texts, [0, 1, 2, 3])
    over.elements.append(PredElement(unit_id="x", block_type="text_block", category="text_block",
                                     page=1, seq=9, text="spurious extra paragraph text"))
    do = mpdb.score_text_diag(mpdb.match_text(over, gold))
    assert do["n_pred"] == 5 and do["precision"] < 1.0 and do["recall"] == pytest.approx(1.0)


def test_leading_text_empty_page_does_not_shift_alignment():
    """A page that has content but NO text on one side (a cover / full-image page, or a page whose text
    the extractor silently dropped) must not shift every later page's rank. With a page_universe that
    covers all physical pages, the 'beta' block on page 2 still matches its gold twin. Without the fix,
    ranks are built over text-bearing pages only, so gold [p1,p2]->ranks[0,1] but pred [p2]->rank[0] and
    'beta' is compared against page-1 'alpha' -> the true match is lost (n_true_matches 1 -> 0)."""
    gold = GoldDoc(
        text_elems=[{"page_id": 1, "order": 1, "category": "text_block", "text": "alpha first block"},
                    {"page_id": 2, "order": 2, "category": "text_block", "text": "beta second block"}],
        page_universe=[1, 2])
    # pred has text ONLY on page 2; page 1 exists in the universe (cover, furniture-only) but no text.
    pred = PredDoc(
        elements=[PredElement(unit_id="u", block_type="text_block", category="text_block",
                              page=2, seq=1, text="beta second block")],
        page_universe=[1, 2])
    diag = mpdb.score_text_diag(mpdb.match_text(pred, gold))
    assert diag["n_gold"] == 2 and diag["n_pred"] == 1
    assert diag["n_true_matches"] == 1  # 'beta' matched to its page-2 twin, not to page-1 'alpha'


def test_page_present_on_one_side_only_does_not_shift_aligned_pages():
    """A page listed in ONE side's universe but empty on both (an interior gap) must NOT shift the ranks
    of the pages both sides share — the textBook_99585330 regression, where pred's universe listed a
    physical page gold never annotated and desynced every later page. gamma is on physical page 3 both
    sides; pred additionally lists an empty physical page 2 that gold skipped entirely."""
    gold = GoldDoc(
        text_elems=[{"page_id": 1, "order": 1, "category": "text_block", "text": "alpha block one"},
                    {"page_id": 3, "order": 1, "category": "text_block", "text": "gamma block three"}],
        page_universe=[1, 3])                       # gold has NOTHING on physical page 2
    pred = PredDoc(
        elements=[PredElement(unit_id="a", block_type="text_block", category="text_block",
                              page=0, seq=1, text="alpha block one"),
                  PredElement(unit_id="c", block_type="text_block", category="text_block",
                              page=2, seq=1, text="gamma block three")],
        page_universe=[0, 1, 2])                    # pred universe LISTS physical page 2 (ocr1), no text
    diag = mpdb.score_text_diag(mpdb.match_text(pred, gold))
    assert diag["n_true_matches"] == 2  # alpha & gamma still align despite pred's extra empty page


def test_builders_populate_page_universe():
    """build_gold_doc covers every page that has any layout det; build_pred_doc anchors to the doc's OCR
    layout pages, falling back to the pages present in the units when no layout dir exists ('testdoc')."""
    assert mpdb.build_gold_doc(_gold(with_truncation=True)).page_universe == [1, 2]
    assert mpdb.build_gold_doc(_gold()).page_universe == [1]
    assert mpdb.build_pred_doc(_fd(), "testdoc").page_universe == [1]


def test_score_merged_text_and_table():
    g = mpdb.build_gold_doc(_gold(with_truncation=True))
    p = PredDoc(elements=[PredElement(unit_id="u", block_type="text_block", category="text_block",
                                      page=1, seq=1, text="part one part two", merged=True)])
    assert mpdb.score_merged_text(p, g)["score"] == pytest.approx(1.0)
    # no gold merges -> None
    assert mpdb.score_merged_text(p, mpdb.build_gold_doc(_gold())) is None
    # merged table
    gt = GoldDoc(merged_tables=[{"order": 1, "html": _TABLE_HTML, "is_merged": "table"}])
    pt = PredDoc(elements=[PredElement(unit_id="t", block_type="table", category="table",
                                       page=1, seq=1, html=_TABLE_HTML, merged=True)])
    assert mpdb.score_merged_table(pt, gt)["teds"] == pytest.approx(1.0)


def test_merged_text_credits_within_page_inline_merge():
    """A within-page (column) merge is emitted inline as ONE single-page block (merged=False).
    The reconstruction-fidelity metric must credit it (the old cross-page-span-only proxy scored 0),
    while a genuine column-split (two half-blocks) must NOT be credited."""
    g = mpdb.build_gold_doc(_gold(with_truncation=True))  # gold merged text = "part one part two"
    inline = PredDoc(elements=[PredElement(unit_id="u", block_type="text_block",
                                           category="text_block", page=1, seq=1,
                                           text="part one part two", merged=False)])
    res = mpdb.score_merged_text(inline, g)
    assert res["score"] == pytest.approx(1.0)
    assert res["n_reconstructed"] == 1
    assert res["n_pred_merges"] == 0  # diagnostic: no cross-page span, yet still credited
    split = PredDoc(elements=[
        PredElement(unit_id="a", block_type="text_block", category="text_block", page=1, seq=1,
                    text="part one", merged=False),
        PredElement(unit_id="b", block_type="text_block", category="text_block", page=1, seq=2,
                    text="part two", merged=False)])
    assert mpdb.score_merged_text(split, g)["score"] < 0.8


def test_merged_table_credits_single_page_table():
    """A same-page column-split table emitted as ONE table (merged=False) must be credited."""
    gt = GoldDoc(merged_tables=[{"order": 1, "html": _TABLE_HTML, "is_merged": "table"}])
    pt = PredDoc(elements=[PredElement(unit_id="t", block_type="table", category="table",
                                       page=1, seq=1, html=_TABLE_HTML, merged=False)])
    res = mpdb.score_merged_table(pt, gt)
    assert res["teds"] == pytest.approx(1.0)
    assert res["n_reconstructed"] == 1 and res["n_pred_merges"] == 0


def test_score_relation_optional():
    # goverment-like: two truncated tables merged; best-effort, must not raise
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


# --------------------------------------------------------------------------- 5. evaluate_doc
def test_evaluate_doc_end_to_end():
    res = mpdb.evaluate_doc("testdoc", _fd(), _gold(with_truncation=True))
    assert res["doc"] == "testdoc"
    for key in ("text", "equation", "equation_diag", "table", "heads", "reading_order", "merged_text",
                "merged_table", "counts"):
        assert key in res
    # every present numeric primary is in [0,1]
    for _name, path in mpdb._PRIMARY:
        v = mpdb._dig(res, path)
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{path} out of range: {v}"


# --------------------------------------------------------------------------- 6. report render
def test_render_weak_spots():
    results = {"run": "r", "skipped_no_gold": [], "models": {"gemma": {
        "per_doc": [mpdb.evaluate_doc("testdoc", _fd(), _gold())],
        "aggregate": {"overall": {k: 0.4 for k, _ in mpdb._METRIC_LABELS},
                      "by_domain": {"book": {k: 0.4 for k, _ in mpdb._METRIC_LABELS}},
                      "n_docs": 1}}}}
    md = mpdb.render_weak_spots(results, {"testdoc": "book"})
    assert "Model: gemma" in md and "book" in md and "{{" not in md
    assert "| Domain |" in md


# --------------------------------------------------------------------------- 7. run wiring
def test_evaluate_run_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(mpdb, "load_gold_index", lambda: {})
    monkeypatch.setattr(mpdb, "_domain_map", lambda stems: {})
    res = mpdb.evaluate_run(tmp_path)
    assert res["models"] == {}
    out = tmp_path / "_mpdocbench"
    assert (out / "metrics.json").is_file() and (out / "weak_spots.md").is_file()
    assert "No scored documents" in (out / "weak_spots.md").read_text(encoding="utf-8")


def test_evaluate_run_scores_synthetic_doc(tmp_path, monkeypatch):
    # a fake multi-model run: runs/<run>/testdoc/gemma/final_document.json
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


# --------------------------------------------------------------------------- 8. paper-faithful table
# matching (Hungarian best-match) + text-redirect of unmatched pred tables + our count diagnostic.
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
    """build_pred_doc stashes the flattened cell text on a table element so the redirect can reuse it."""
    tbl = next(e for e in mpdb.build_pred_doc(_fd(), "testdoc").elements if e.block_type == "table")
    assert tbl.table_text == table_to_text({"table": {"headers": ["A", "B"], "rows": [["1", "2"]]}})
    assert tbl.table_text  # non-empty ("A B 1 2")


def test_match_tables_is_order_independent():
    """Hungarian best-match pairs each gold with its true twin regardless of prediction order — the
    fragility of the old index-order matching (perfectly-extracted-but-swapped -> ~0) is gone."""
    g = GoldDoc(tables=[{"html": _TABLE_HTML}, {"html": _TABLE_HTML_2}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML_2, 1), _tbl_elem("p1", _TABLE_HTML, 2)])  # swapped
    m = mpdb.match_tables(p, g)
    r = mpdb.score_tables(p, g, m)
    assert r["teds"] == pytest.approx(1.0) and r["n_matched"] == 2 and not m.unmatched_pred


def test_match_tables_missed_gold_scores_zero():
    """A gold table with no prediction counts as 0 (recall penalty), as in the paper."""
    g = GoldDoc(tables=[{"html": _TABLE_HTML}, {"html": _TABLE_HTML_2}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML, 1)])  # only the first gold is predicted
    m = mpdb.match_tables(p, g)
    r = mpdb.score_tables(p, g, m)
    assert r["n_gold"] == 2 and r["n_matched"] == 1
    assert r["teds"] == pytest.approx(0.5, abs=0.05)  # (1.0 + 0.0) / 2


def test_apply_redirect_when_gold_has_no_tables():
    """Prospectus-TOC case: gold encodes it as text, we emitted a table. Zero gold tables -> every pred
    table is unmatched -> redirected to the text stream so the (whole-string) text metric credits it."""
    gold = mpdb.build_gold_doc(_gold_without_tables())
    assert gold.tables == []
    pred = mpdb.build_pred_doc(_fd(), "testdoc")
    before = mpdb.score_text(pred, gold)["pred_chars"]  # table text excluded pre-redirect
    m = mpdb.match_tables(pred, gold)
    assert len(m.unmatched_pred) == 1 and m.n_gold == 0
    mpdb.apply_table_text_redirect(pred, m)
    assert not any(e.block_type == "table" for e in pred.elements)
    demoted = next(e for e in pred.elements if "1" in e.text and "2" in e.text)
    assert demoted.html == "" and demoted.block_type == "text_block" and demoted.category == "text_block"
    assert mpdb.score_text(pred, gold)["pred_chars"] > before


def test_redirect_keeps_matched_table_and_redirects_extra():
    """Over-prediction (P>G): the pred table that wins the gold stays a table (still scored); the extra
    'table' loses the assignment and its text is redirected — headline table TEDS stays perfect."""
    g = GoldDoc(tables=[{"html": _TABLE_HTML}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML, 1, "A B 1 2"),
                          _tbl_elem("p1", "<table><tr><td>Introduction ..... 3</td></tr></table>", 2,
                                    "Introduction ..... 3")])  # spurious extra
    m = mpdb.match_tables(p, g)
    mpdb.apply_table_text_redirect(p, m)
    kept = [e for e in p.elements if e.block_type == "table"]
    assert len(kept) == 1 and "<td>A</td>" in kept[0].html
    assert any("Introduction" in e.text for e in p.elements if e.block_type == "text_block")
    assert mpdb.score_tables(p, g, m)["teds"] == pytest.approx(1.0)


def test_table_diag_flags_overprediction():
    """Diagnostic: an extra predicted table pulls precision below 1 (recall stays 1)."""
    g = GoldDoc(tables=[{"html": _TABLE_HTML}])
    p = PredDoc(elements=[_tbl_elem("p0", _TABLE_HTML, 1),
                          _tbl_elem("p1", _UNRELATED_TABLE, 2)])  # spurious
    d = mpdb.score_table_diag(mpdb.match_tables(p, g))
    assert d["n_pred"] == 2 and d["n_true_matches"] == 1
    assert d["precision"] == pytest.approx(0.5) and d["recall"] == pytest.approx(1.0)


def test_table_diag_flags_equalcount_misclassification():
    """The case the headline hides: 1 gold, 1 pred but the pred is a non-table (TEDS < 0.5). Hungarian
    still ASSIGNS it (no threshold) so the headline counts its low TEDS; the diagnostic F1 -> 0."""
    g = GoldDoc(tables=[{"html": _TABLE_HTML}])
    p = PredDoc(elements=[_tbl_elem("p0", _UNRELATED_TABLE, 1)])
    m = mpdb.match_tables(p, g)
    assert m.n_pred == 1 and len(m.matched_pred_idxs) == 1  # assigned despite being a poor match
    assert mpdb.score_tables(p, g, m)["teds"] < 0.5        # headline counts the low TEDS
    d = mpdb.score_table_diag(m)
    assert d["n_true_matches"] == 0 and d["f1"] == 0.0     # diagnostic flags it


def test_table_diag_none_when_no_tables_either_side():
    assert mpdb.score_table_diag(mpdb.match_tables(PredDoc(), GoldDoc())) is None


def test_evaluate_doc_redirects_unmatched_table_into_text():
    """End-to-end: with no gold table, evaluate_doc redirects the table's text into the text score, the
    table metric stays None, and the diagnostic reports the over-prediction (precision 0)."""
    res = mpdb.evaluate_doc("testdoc", _fd(), _gold_without_tables())
    assert res["table"] is None
    assert res["text"]["pred_chars"] > 0 and res["text"]["score"] > 0.0
    assert res["table_diag"]["precision"] == 0.0 and res["table_diag"]["n_pred"] == 1


def test_find_struggles_flags_headings_and_table_overprediction():
    """The generated report's narrative now surfaces the true #1/#3 weak spots: heading-tree quality
    and table over-prediction (via table_diag), not just table TEDS / reading-order / text / merge."""
    per_doc = [{"doc": "d1", "heads": {"score": 0.20, "n_gold_titles": 10, "n_pred_heads": 14},
                "table_diag": {"precision": 0.0, "n_gold": 0, "n_pred": 5}}]
    out = " ".join(mpdb._find_struggles(per_doc, {"d1": "magazine"}))
    assert "heading-tree 0.20" in out and "gold titles) — heading level/nesting" in out
    assert "over-predicts tables — 5 pred vs 0 gold" in out
    # a clean doc (good heads, no over-prediction) flags neither
    clean = [{"doc": "d2", "heads": {"score": 0.9, "n_gold_titles": 10, "n_pred_heads": 10},
              "table_diag": {"precision": 1.0, "n_gold": 2, "n_pred": 2}}]
    assert mpdb._find_struggles(clean, {"d2": "paper"}) == []

"""Page under-production detector (Track 2 page-drop guard).

A page is "under-produced" when the extractor bailed out mid-page: the region analyzer found many
regions and the OCR has real text, but the extractor emitted only a couple of near-empty blocks that
capture almost none of the page. The detector is a pure function of (region count, block texts, OCR
text) so it is easy to test in isolation and reuse from the orchestrator."""

from src.stages.underproduction import analyze_underproduction


def test_flags_bailout_page():
    # goverment_10e7766e pg2: 18 regions, one stray "he adds." fragment for a full page of text.
    r = analyze_underproduction(18, ["he adds."], "word " * 600)
    assert r["under_produced"] is True
    assert r["n_regions"] == 18 and r["n_blocks"] == 1
    assert r["coverage"] < 0.05


def test_flags_single_empty_table_for_dense_page():
    # paper_d76efc79 pg8: 44 regions -> one empty table block.
    r = analyze_underproduction(44, [""], "content " * 700)
    assert r["under_produced"] is True


def test_spares_proportionate_page():
    texts = [f"paragraph number {i} with plenty of real words here " * 4 for i in range(8)]
    r = analyze_underproduction(8, texts, " ".join(texts))
    assert r["under_produced"] is False


def test_spares_near_blank_page():
    # A page-number/furniture-only page: 1 region, tiny OCR, 0 content blocks -> not a bail-out.
    r = analyze_underproduction(1, [], "12")
    assert r["under_produced"] is False


def test_spares_single_big_block_that_covers_the_page():
    # A legitimate one-big-table / one-block page: few blocks but coverage ~1.0.
    ocr = "cell value " * 200
    r = analyze_underproduction(6, [ocr], ocr)
    assert r["under_produced"] is False


def test_spares_oversplit_regions_when_coverage_is_high():
    # Region analyzer over-split into 8 regions but 2 blocks still capture ~all the text.
    ocr = "alpha " * 120 + "beta " * 120
    r = analyze_underproduction(8, ["alpha " * 120, "beta " * 120], ocr)
    assert r["under_produced"] is False


def test_spares_moderate_undersegmentation_above_coverage_threshold():
    # goverment_446784ca pg4 (qwen): a dense directory, 67 regions -> 13 real blocks at ~45% coverage.
    # That is MODERATE under-segmentation (Track 3), NOT a page-drop bail-out (real bail-outs sit at
    # <=0.11 coverage) — the guard must leave it to the de-merge track, not flag it.
    ocr = "word " * 1000                         # 4000 normalized chars
    blocks = ["word " * 35 for _ in range(13)]   # ~1820 chars -> coverage ~0.455
    r = analyze_underproduction(67, blocks, ocr)
    assert 0.3 < r["coverage"] < 0.6
    assert r["under_produced"] is False


def test_spares_huge_dense_page_with_many_blocks():
    # magazine_22066f34 pg8: a 34k-char spread with 25 blocks and low coverage is NOT a bail-out —
    # 25 blocks far exceeds the block cap, so it must not be flagged.
    r = analyze_underproduction(29, ["block text"] * 25, "x " * 17000)
    assert r["under_produced"] is False

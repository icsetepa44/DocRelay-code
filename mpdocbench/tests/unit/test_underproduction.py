from src.stages.underproduction import analyze_underproduction


def test_flags_bailout_page():
    r = analyze_underproduction(18, ["he adds."], "word " * 600)
    assert r["under_produced"] is True
    assert r["n_regions"] == 18 and r["n_blocks"] == 1
    assert r["coverage"] < 0.05


def test_flags_single_empty_table_for_dense_page():
    r = analyze_underproduction(44, [""], "content " * 700)
    assert r["under_produced"] is True


def test_spares_proportionate_page():
    texts = [f"paragraph number {i} with plenty of real words here " * 4 for i in range(8)]
    r = analyze_underproduction(8, texts, " ".join(texts))
    assert r["under_produced"] is False


def test_spares_near_blank_page():
    r = analyze_underproduction(1, [], "12")
    assert r["under_produced"] is False


def test_spares_single_big_block_that_covers_the_page():
    ocr = "cell value " * 200
    r = analyze_underproduction(6, [ocr], ocr)
    assert r["under_produced"] is False


def test_spares_oversplit_regions_when_coverage_is_high():
    ocr = "alpha " * 120 + "beta " * 120
    r = analyze_underproduction(8, ["alpha " * 120, "beta " * 120], ocr)
    assert r["under_produced"] is False


def test_spares_moderate_undersegmentation_above_coverage_threshold():
    ocr = "word " * 1000
    blocks = ["word " * 35 for _ in range(13)]
    r = analyze_underproduction(67, blocks, ocr)
    assert 0.3 < r["coverage"] < 0.6
    assert r["under_produced"] is False


def test_spares_huge_dense_page_with_many_blocks():
    r = analyze_underproduction(29, ["block text"] * 25, "x " * 17000)
    assert r["under_produced"] is False

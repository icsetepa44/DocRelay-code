import json
from pathlib import Path

import pytest

import run as runner
from src.core import config
from src.ocr import chandra_adapter, preflight


DEFAULT_ROOT = config.MPDOCBENCH_ROOT / "ocr"


def test_ocr_root_is_the_data_folder_unless_overridden():
    root = (config.PIPELINE.get("inputs") or {}).get("ocr_root")
    expected = (config.PACKAGE_ROOT / root).resolve() if root else DEFAULT_ROOT
    assert config.BENCHMARK_OCR_PARENT == expected


def _make_doc(root: Path, doc: str, pages, marker=True, drop_image=None, drop_layout=False, category=None):
    d = root / category / doc if category else root / doc
    for sub in ("markdown", "layout", "page_images"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(pages):
        (d / "markdown" / f"page_{i:04d}.md").write_text(text, encoding="utf-8")
        if not (drop_layout and i == 0):
            (d / "layout" / f"page_{i:04d}.json").write_text("{}", encoding="utf-8")
        if drop_image != i:
            (d / "page_images" / f"page_{i:04d}.png").write_bytes(b"png")
    if marker:
        (d / f"{doc}.json").write_text(json.dumps({"pages": len(pages)}), encoding="utf-8")


def test_preflight_clean_and_soft_issues(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BENCHMARK_OCR_PARENT", tmp_path)
    _make_doc(tmp_path, "paper_eng_ok", ["A real paragraph of text.", " ", "x" * 50 + ", no" * 80])
    rep = preflight.run_preflight(["paper_eng_ok"], gold_stems={"paper_eng_ok"}, clip_chars=60)
    assert rep["ok"] and rep["summary"]["n_pages"] == 3
    assert rep["summary"]["by_domain"] == {"paper": 1}
    soft = rep["soft"]
    assert [x["page"] for x in soft["near_empty"]] == ["page_0001"]
    assert [x["page"] for x in soft["over_clip"]] == ["page_0002"]
    assert [x["page"] for x in soft["repetition"]] == ["page_0002"]


@pytest.mark.parametrize("kwargs, gold, problem", [
    ({"marker": False}, True, "no completion marker"),
    ({"drop_image": 1}, True, "page count mismatch"),
    ({"drop_layout": True}, True, "page count mismatch"),
    ({}, False, "no gold entry"),
])
def test_preflight_hard_problems_block(tmp_path, monkeypatch, kwargs, gold, problem):
    monkeypatch.setattr(config, "BENCHMARK_OCR_PARENT", tmp_path)
    _make_doc(tmp_path, "book_eng_bad", ["one page of text", "another page"], **kwargs)
    rep = preflight.run_preflight(["book_eng_bad"], gold_stems={"book_eng_bad"} if gold else set())
    assert not rep["ok"]
    assert any(problem in h["problem"] for h in rep["hard"])


def test_nested_category_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BENCHMARK_OCR_PARENT", tmp_path)
    _make_doc(tmp_path, "paper_eng_flat", ["flat page text here"])
    _make_doc(tmp_path, "book_chn_nested", ["嵌套目录中的页面文本。"], category="book")
    _make_doc(tmp_path, "newspaper_chn_unfinished", ["只有一页"], marker=False, category="newspaper")
    (tmp_path / "run_ocr.log").write_text("log", encoding="utf-8")
    assert chandra_adapter.ocr_dir_for("paper_eng_flat") == tmp_path / "paper_eng_flat"
    assert chandra_adapter.ocr_dir_for("book_chn_nested") == tmp_path / "book" / "book_chn_nested"
    assert chandra_adapter.ocr_dir_for("nope_chn_x") == tmp_path / "nope_chn_x"
    assert chandra_adapter.list_available_documents() == ["book_chn_nested", "paper_eng_flat"]
    recs = chandra_adapter.read_page_records("book_chn_nested")
    assert recs[0]["content"]["raw_markdown"] == "嵌套目录中的页面文本。"
    assert Path(recs[0]["page_image_src"]).exists()
    rep = preflight.run_preflight(["book_chn_nested"], gold_stems={"book_chn_nested"})
    assert rep["ok"] and rep["summary"]["by_domain"] == {"book": 1}
    rep = preflight.run_preflight(["newspaper_chn_unfinished"], gold_stems={"newspaper_chn_unfinished"})
    assert not rep["ok"] and rep["hard"][0]["problem"] == "no completion marker"


def test_preflight_missing_folder_is_hard(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BENCHMARK_OCR_PARENT", tmp_path)
    rep = preflight.run_preflight(["nope_eng_x"], gold_stems=None)
    assert not rep["ok"] and rep["hard"][0]["problem"] == "no OCR folder"


def test_doc_progress_postfix():
    assert runner._doc_progress(0, 182, 0) == "docs 0/182 done, 0 errors"
    assert runner._doc_progress(71, 182, 2) == "docs 71/182 done, 2 errors"
    assert runner._doc_progress(182, 182, 0) == "docs 182/182 done, 0 errors"


def test_quiet_is_default():
    assert runner.QUIET is True

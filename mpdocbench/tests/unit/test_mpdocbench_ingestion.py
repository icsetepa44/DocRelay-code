"""MPDocBench ingestion — read layout + page image from the OCR dir, no PDFs.

Hermetic tests build a synthetic mini-OCR doc in tmp and monkeypatch the OCR/data paths.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.contracts.page import PageData
from src.core import config
from src.ocr import chandra_adapter
from src.storage import document_store as ds


def _make_mini_ocr_doc(parent: Path, doc: str) -> dict:
    d = parent / doc
    for sub in ("markdown", "html", "layout", "page_images"):
        (d / sub).mkdir(parents=True)
    (d / "markdown" / "page_0000.md").write_text("# Title\n\nHello world.", encoding="utf-8")
    (d / "html" / "page_0000.html").write_text("<p>Hello world.</p>", encoding="utf-8")
    layout = {"page_num": 0, "page_box": [0, 0, 100, 200],
              "chunks": [{"bbox": [1, 2, 3, 4], "label": "Text", "content": "<p>Hello world.</p>"}]}
    (d / "layout" / "page_0000.json").write_text(json.dumps(layout), encoding="utf-8")
    Image.new("RGB", (10, 20), "white").save(d / "page_images" / "page_0000.png")
    (d / f"{doc}.json").write_text(json.dumps({"file_name": f"{doc}.pdf", "pages": []}),
                                   encoding="utf-8")
    return layout


def test_read_page_records_surfaces_layout_and_page_image(tmp_path, monkeypatch):
    doc = "paper_eng_test01"
    monkeypatch.setattr(config, "BENCHMARK_OCR_PARENT", tmp_path / "ocr")
    _make_mini_ocr_doc(tmp_path / "ocr", doc)
    recs = chandra_adapter.read_page_records(doc)
    assert len(recs) == 1
    r = recs[0]
    assert r["layout"]["page_box"] == [0, 0, 100, 200]
    assert r["layout"]["chunks"][0]["label"] == "Text"
    assert Path(r["page_image_src"]).exists()
    assert r["page_image_src"].replace("\\", "/").endswith("page_images/page_0000.png")


def test_ensure_page_assets_uses_ocr_image_and_writes_no_pagepng(tmp_path, monkeypatch):
    doc = "paper_eng_test01"
    monkeypatch.setattr(config, "BENCHMARK_OCR_PARENT", tmp_path / "ocr")
    monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_path / "data" / "documents")
    _make_mini_ocr_doc(tmp_path / "ocr", doc)
    rec = chandra_adapter.read_page_records(doc)[0]
    pd = ds.ensure_page_assets(rec)
    assert isinstance(pd, PageData)
    assert pd.layout["chunks"]                                    # layout carried onto PageData
    assert pd.page_image == rec["page_image_src"]                 # points straight at the OCR image
    assert Path(pd.page_image).exists()
    page_dir = config.DOCUMENTS_DIR / doc / "pages" / "0000"
    assert (page_dir / "page_data.json").exists()
    assert (page_dir / "chandra.md").exists()
    assert not (page_dir / "page.png").exists()                   # no PDF render, no copy


def test_ensure_page_assets_missing_image_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_path / "data" / "documents")
    rec = {"document_id": "d", "page_number": 0, "content": {"raw_markdown": "x"},
           "layout": {}, "page_image_src": str(tmp_path / "does_not_exist.png")}
    with pytest.raises(FileNotFoundError):
        ds.ensure_page_assets(rec)


def test_pagedata_layout_roundtrips():
    pd = PageData(document_id="d", page_number=0, page_image="x.png", markdown="m", html="h",
                  layout={"page_box": [0, 0, 1, 1], "chunks": [{"label": "Text"}]})
    assert PageData.from_dict(pd.to_dict()).layout == pd.layout
    # default is an empty dict when absent (backward-compatible)
    assert PageData.from_dict({"document_id": "d", "page_number": 0}).layout == {}



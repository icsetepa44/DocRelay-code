import pytest

from src.core import config
from src.ocr import chandra_adapter as ocr

DOC = "20250310_Customs_Risk_Management_Rules_2025"


@pytest.mark.skipif(
    not (config.BENCHMARK_OCR_PARENT / DOC).exists(),
    reason="benchmark OCR output not present",
)
def test_read_page_records_sorted_and_build_page_data():
    records = ocr.read_page_records(DOC)
    assert len(records) >= 1
    nums = [int(r["page_number"]) for r in records]
    assert nums == sorted(nums)           # sorted by page_number
    assert nums[0] == 0                    # 0-indexed

    pd = ocr.build_page_data(records[0], page_image="fake/page.png")
    assert pd.document_id == DOC
    assert pd.page_number == 0
    assert pd.page_image == "fake/page.png"
    assert isinstance(pd.markdown, str) and isinstance(pd.html, str)
    assert pd.markdown  # non-empty OCR text

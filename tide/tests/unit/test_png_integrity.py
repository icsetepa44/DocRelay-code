from PIL import Image

from src.storage.document_store import _png_complete


def _write_png(path):
    Image.new("RGB", (8, 8), (123, 45, 67)).save(path, "PNG")


def test_complete_png_is_detected_valid(tmp_path):
    p = tmp_path / "page.png"
    _write_png(p)
    assert _png_complete(p) is True


def test_torn_write_is_detected_incomplete(tmp_path):
    """A save interrupted mid-write leaves the file present but missing its IEND chunk."""
    p = tmp_path / "page.png"
    _write_png(p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # simulate a torn write
    assert _png_complete(p) is False


def test_missing_file_is_incomplete(tmp_path):
    assert _png_complete(tmp_path / "nope.png") is False

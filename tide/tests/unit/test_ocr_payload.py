"""Tests for the 'mixed' OCR format: markdown prose + inline HTML tables, order preserved.

Covers ``mixed_ocr_text`` (the parser-based normalizer) and the ``page_text_payload``
'mixed' branch that feeds every LLM stage.
"""

from types import SimpleNamespace

from src.stages._common import mixed_ocr_text, page_text_payload

TABLE = (
    "<table>\n<tr><td>1</td><td>Name</td></tr>\n"
    "<tr><td>2</td><td>Age</td></tr>\n</table>"
)


def test_prose_only_is_unchanged():
    md = "#### Rule 4\n\nThe Commissioner may allow the goods.\n\n- point ka\n- point kha"
    assert mixed_ocr_text(md) == md  # fast path: no table, returned verbatim


def test_prose_plus_table_keeps_prose_table_and_order():
    md = f"Intro prose before the form.\n\n{TABLE}\n\nOutro prose after the form."
    out = mixed_ocr_text(md)
    assert "Intro prose before the form." in out
    assert "Outro prose after the form." in out
    assert "<table>" in out and "</table>" in out
    assert "<td>Name</td>" in out
    # reading order preserved: prose -> table -> prose
    assert out.find("Intro prose") < out.find("<table>") < out.find("Outro prose")


def test_table_with_attributes_and_thead_tbody_preserved():
    md = (
        "Schedule below:\n\n"
        '<table border="1"><thead><tr><th>Item</th><th>Rate</th></tr></thead>'
        "<tbody><tr><td>TV</td><td>10%</td></tr></tbody></table>\n\ndone."
    )
    out = mixed_ocr_text(md)
    assert "<thead>" in out and "<tbody>" in out
    assert "<th>Item</th>" in out
    assert 'border="1"' in out  # attributes survive


def test_two_tables_with_prose_between_keep_order():
    md = f"First para.\n\n{TABLE}\n\nMiddle para.\n\n{TABLE}\n\nLast para."
    out = mixed_ocr_text(md)
    i_first = out.find("First para")
    i_t1 = out.find("<table>")
    i_mid = out.find("Middle para")
    i_t2 = out.find("<table>", i_t1 + 1)
    i_last = out.find("Last para")
    assert -1 not in (i_first, i_t1, i_mid, i_t2, i_last)
    assert i_first < i_t1 < i_mid < i_t2 < i_last
    assert out.count("<table>") == 2


def test_malformed_unclosed_table_does_not_crash_and_yields_table():
    md = "Before.\n\n<table><tr><td>x</td><td>y</td></tr>\n\nAfter text with no close."
    out = mixed_ocr_text(md)  # must not raise
    assert "<table>" in out          # bs4 emits a (normalized) table
    assert "Before." in out


def test_payload_mixed_returns_single_markdown_key():
    page = SimpleNamespace(
        markdown=f"Prose.\n\n{TABLE}\n\nMore prose.",
        html="<p>Prose.</p>" + TABLE + "<p>More prose.</p>",
    )
    out = page_text_payload(page, "mixed")
    assert set(out.keys()) == {"markdown"}          # no separate 'html' key
    assert "<table>" in out["markdown"]              # table kept inline
    assert "Prose." in out["markdown"]


def test_payload_existing_modes_unchanged():
    page = SimpleNamespace(markdown="MD", html="HTML")
    assert page_text_payload(page, "markdown") == {"markdown": "MD"}
    assert page_text_payload(page, "html") == {"html": "HTML"}
    assert page_text_payload(page, "both") == {"markdown": "MD", "html": "HTML"}
    # unknown value guards back to 'both'
    assert page_text_payload(page, "bogus") == {"markdown": "MD", "html": "HTML"}

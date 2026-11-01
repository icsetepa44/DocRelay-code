from src.core import config
from src.llm.prompt_loader import load_prompt


def test_extractor_heading_level_convention_is_explicit():
    p = load_prompt("extractor")
    assert "ALWAYS include `level`" in p
    assert "level 1 = the single overall DOCUMENT" in p
    assert "level 2 = a top-level section heading" in p
    assert "EVEN on later pages" in p


def test_extractor_rule12_gives_prose_parity_with_tables():
    p = load_prompt("extractor")
    assert "PROSE parity" in p
    assert "never silently drop a cut paragraph" in p


def test_continuation_judge_has_dedicated_prose_clause():
    p = load_prompt("continuation_judge")
    assert "open PROSE PARAGRAPH" in p
    assert "open_text" in p and "mid-sentence" in p


def test_extractor_column_reading_order_is_strengthened():
    p = load_prompt("extractor")
    assert "read each column FULLY top-to-bottom before moving to the next column" in p


def test_extractor_same_page_column_merge_rule():
    p = load_prompt("extractor")
    assert "SAME-PAGE COLUMN CONTINUATION" in p
    assert "boundary is NOT an element boundary" in p
    assert 'do NOT mark it "continues"' in p


def test_domain_profiles_distinguish_column_break_from_page_break():
    for dom in ("magazine", "newspaper"):
        g = config.domain_profile(dom)["guidance"]
        assert "COLUMN break" in g and "PAGE bottom" in g
        assert "page or column break" not in g


def test_extractor_same_page_merge_uses_sentence_not_column_gap_cue():
    p = load_prompt("extractor")
    assert "DECIDE BY THE SENTENCE, NOT THE COLUMN GAP" in p
    assert "ends WITHOUT terminal" in p and "err toward merging a mid-sentence" in p


def test_extractor_cross_page_continues_restricted_to_last_column():
    p = load_prompt("extractor")
    assert "LAST column may continue onto the NEXT PAGE" in p

"""Systematic base-prompt fixes.

Content-presence guards for three base-prompt changes that address weak spots found during development —
heading level/nesting, cross-page PROSE continuation, and within-page MULTI-COLUMN merge. Hermetic: they only read prompt text, no LLM/scored run.
They exist so a future prompt refactor can't silently drop these rules."""

from src.core import config
from src.llm.prompt_loader import load_prompt


# --------------------------------------------------------------- Fix 1: heading level / nesting
def test_extractor_heading_level_convention_is_explicit():
    p = load_prompt("extractor")
    # level is mandatory (missing level flattens the tree) ...
    assert "ALWAYS include `level`" in p
    # ... and follows gold's convention: one document-title root, sections one level below it,
    # marked level 2 even on later pages where the title isn't visible.
    assert "level 1 = the single overall DOCUMENT" in p
    assert "level 2 = a top-level section heading" in p
    assert "EVEN on later pages" in p


# --------------------------------------------------------------- Fix 2: cross-page prose continuation
def test_extractor_rule12_gives_prose_parity_with_tables():
    p = load_prompt("extractor")
    assert "PROSE parity" in p
    # a cut paragraph must not be silently closed; borderline goes to the judge via "uncertain"
    assert "never silently drop a cut paragraph" in p


def test_continuation_judge_has_dedicated_prose_clause():
    p = load_prompt("continuation_judge")
    assert "open PROSE PARAGRAPH" in p
    # it keys off open_text ending mid-sentence, mirroring the existing table clause
    assert "open_text" in p and "mid-sentence" in p


# --------------------------------------------------------------- Fix 3: within-page multi-column
def test_extractor_column_reading_order_is_strengthened():
    p = load_prompt("extractor")
    assert "read each column FULLY top-to-bottom before moving to the next column" in p


def test_extractor_same_page_column_merge_rule():
    p = load_prompt("extractor")
    assert "SAME-PAGE COLUMN CONTINUATION" in p
    assert "boundary is NOT an element boundary" in p
    # the same-page merge is inline — explicitly NOT the cross-page stitch mechanism
    assert 'do NOT mark it "continues"' in p


def test_domain_profiles_distinguish_column_break_from_page_break():
    # magazine/newspaper guidance must route a COLUMN break to same-page merge and a PAGE break to
    # "continues". The YAML folded scalar collapses newlines to spaces, so check the folded string.
    for dom in ("magazine", "newspaper"):
        g = config.domain_profile(dom)["guidance"]
        assert "COLUMN break" in g and "PAGE bottom" in g
        # the old "unfinished at a page or column break as continuing" wording is gone
        assert "page or column break" not in g


# ------------------------------------ Precision rules
# A few newspaper column-splits + over-eager cross-page "continues". These guard the two prompt rules that
# address them.
def test_extractor_same_page_merge_uses_sentence_not_column_gap_cue():
    p = load_prompt("extractor")
    assert "DECIDE BY THE SENTENCE, NOT THE COLUMN GAP" in p
    # the concrete cue: mid-sentence column-bottom + lower-case next column => one paragraph
    assert "ends WITHOUT terminal" in p and "err toward merging a mid-sentence" in p


def test_extractor_cross_page_continues_restricted_to_last_column():
    p = load_prompt("extractor")
    # only the page's LAST column may continue onto the next page; an earlier column continues inline
    assert "LAST column may continue onto the NEXT PAGE" in p

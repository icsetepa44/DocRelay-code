import random

import pytest

from src.core import config
from src.eval import evaluate

needs_official = pytest.mark.skipif(not (config.MPDOCBENCH_REPO / "dataset" / "end2end_dataset.py").is_file(),
                                    reason="official MPDocBench code not downloaded (data/mpdocbench/repo/MPDocBench)")


def _gold(*clauses, tables=()):
    return {"items": [{"type": "clause", "text": c} for c in clauses], "tables": list(tables),
            "full_text": " ".join(clauses), "stats": {},
            "raw": {"parsed_text": [{"type": "clause", "text": c, "text_id": f"c{i}", "page_start": 1, "page_end": 1}
                                    for i, c in enumerate(clauses)],
                    "tables": [dict(t, parent_text_id="c0") for t in tables]}}


def _pred(*clauses, tables=()):
    return {"contents": [{"type": "clause", "text": c, "full_text": c} for c in clauses],
            "tables": [dict(t, parent_index=0) for t in tables], "adapter_stats": {}}


def test_levenshtein():
    assert evaluate._levenshtein("kitten", "sitting") == 3
    assert evaluate._levenshtein([], [1, 2]) == 2


def test_levenshtein_matches_exact_dp_on_random_bangla_english():
    rng = random.Random(7)
    alphabet = list("abcde ") + list("কখগঘঙচছজ্ািীুেো ")
    for n in (0, 1, 5, 40, 120, 520, 700):
        for _ in range(3):
            a = "".join(rng.choice(alphabet) for _ in range(n))
            b = "".join(rng.choice(alphabet) for _ in range(rng.randint(max(0, n - 30), n + 30)))
            assert evaluate._levenshtein(a, b) == evaluate._levenshtein_dp(a, b)


def test_levenshtein_on_token_lists():
    a = "এই প্রজ্ঞাপন অবিলম্বে কার্যকর হইবে".split() * 120
    b = list(a)
    b[3], b[400] = "ভুল", "শব্দ"
    del b[100]
    assert evaluate._levenshtein(a, b) == evaluate._levenshtein_dp(a, b) == 3


def test_official_sample_and_pred_markdown_shapes():
    raw = {"parsed_text": [
        {"type": "clause", "text": "1. Goods [table_1] are exempt.\n\n(a) subject to rules", "text_id": "c0",
         "page_start": 1},
        {"type": "metadata", "metadata_category": "page", "text": "Page 2", "text_id": "m0", "page_start": 2}],
        "tables": [{"table_id": "table_1", "parent_text_id": "c0", "headers": ["HS"], "rows": [["01"]],
                    "page_start": 1}]}
    dets = evaluate.official_sample("d", raw)["layout_dets"]
    assert [x["category_type"] for x in dets] == ["text_block", "page_number", "table"]
    assert dets[0]["text"] == "1. Goods   are exempt.\n(a) subject to rules"
    assert dets[2]["html"] == "<table><tr><td>HS</td></tr><tr><td>01</td></tr></table>" and dets[2]["order"] == 1.5
    md = evaluate.pred_markdown({"contents": [{"text": "A\n\nB"}, {"text": "C"}],
                                 "tables": [{"parent_index": 0, "headers": ["x"], "rows": []},
                                            {"parent_index": None, "rows": [["y"]]}]})
    assert md == "A\nB\n\n<table><tr><td>x</td></tr></table>\n\nC\n\n<table><tr><td>y</td></tr></table>"


@needs_official
def test_evaluate_pair_returns_only_the_paper_fields():
    m = evaluate.evaluate_pair(
        _gold("The applicant shall submit the required documents within thirty days."),
        _pred("The applicant shall submit the required documents within thirty days."))
    assert set(m) == {"gold_clauses", "gold_metadata", "gold_tables", "text_edit", "table_teds"}
    assert m["text_edit"]["edit_dist"] == 0.0 and m["gold_clauses"] == 1 and m["gold_tables"] == 0


@needs_official
def test_official_text_edit_counts_typos():
    m = evaluate.evaluate_pair(
        _gold("The applicant shall submit the required documents within thirty days."),
        _pred("The applicant shall submit the requred documnts within thirty days."))
    assert 0.0 < m["text_edit"]["edit_dist"] < 0.1 and m["text_edit"]["n_blocks"] == 1


_TABLE_KEYS = ("table_teds",)
_TABLE = {"headers": ["h1", "h2"], "rows": [["a", "b"], ["c", "d"]]}


@needs_official
def test_structural_table_keys_none_without_tables():
    m = evaluate.evaluate_pair(_gold("A plain clause with no tables."), _pred("A plain clause with no tables."))
    for k in _TABLE_KEYS:
        assert k in m and m[k] is None


@needs_official
def test_structural_table_keys_perfect_on_identical_table():
    m = evaluate.evaluate_pair(_gold("see the schedule", tables=[_TABLE]), _pred("see the schedule", tables=[_TABLE]))
    for k in _TABLE_KEYS:
        assert m[k] == 1.0
    assert m["gold_tables"] == 1


@needs_official
def test_structural_table_keys_drop_when_pred_table_missing():
    m = evaluate.evaluate_pair(_gold("see the schedule", tables=[_TABLE]), _pred("see the schedule"))
    for k in _TABLE_KEYS:
        assert m[k] == 0.0

"""Unit tests for the P3 structural table metrics (TEDS/TEDS-Struct/GriTS-Con/GriTS-Top).

All deterministic, no LLM. Hand-computable small grids pin the exact behaviour: content vs structure
separation, shape penalties, table matching, the no-tables None case, HTML/markdown fallback parsing,
and scipy-vs-greedy agreement.
"""

import pytest

from src.eval import table_structure as ts

A = [["h1", "h2", "h3"], ["a", "b", "c"], ["d", "e", "f"]]


def test_identical_grid_scores_one():
    assert ts.teds(A, A) == 1.0
    assert ts.teds(A, A, struct_only=True) == 1.0
    con, top = ts.grits(A, A)
    assert round(con, 6) == 1.0 and round(top, 6) == 1.0


def test_cell_text_diff_hits_content_not_structure():
    B = [["h1", "h2", "h3"], ["a", "b", "c"], ["d", "e", "ZZZ"]]
    assert ts.teds(A, B) < 1.0                      # content-aware TEDS notices
    assert ts.teds(A, B, struct_only=True) == 1.0   # structure identical
    con, top = ts.grits(A, B)
    assert con < 1.0 and round(top, 6) == 1.0


def test_missing_row_penalizes_structure():
    C = [["h1", "h2", "h3"], ["a", "b", "c"]]        # one row short
    assert ts.teds(A, C) < 1.0 and ts.teds(A, C, struct_only=True) < 1.0
    _, top = ts.grits(A, C)
    assert abs(top - 2 * (2 * 3) / (9 + 6)) < 1e-9   # 2*min(3,2)*3 / (9+6) = 0.8


def test_extra_column_penalizes_topology():
    D = [["h1", "h2", "h3", "h4"], ["a", "b", "c", "x"], ["d", "e", "f", "y"]]
    _, top = ts.grits(A, D)
    assert abs(top - 2 * (3 * 3) / (9 + 12)) < 1e-9  # 18/21


def test_transpose_is_not_perfect():
    T = [[A[i][j] for i in range(3)] for j in range(3)]  # 3x3 transpose
    assert ts.teds(A, T) < 1.0
    con, _ = ts.grits(A, T)
    assert con < 1.0


def test_to_grid_structured_and_fallbacks():
    assert ts.to_grid({"headers": ["a"], "rows": [["b"], ["c"]]}) == [["a"], ["b"], ["c"]]
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    assert ts.to_grid({"html": html}) == [["a", "b"], ["c", "d"]]
    md = "| a | b |\n| --- | --- |\n| c | d |"
    assert ts.to_grid({"markdown": md}) == [["a", "b"], ["c", "d"]]
    assert ts.to_grid({"markdown": "just prose"}) is None
    assert ts.to_grid({}) is None
    assert ts.to_grid("not a dict") is None


def test_to_grid_rectangularizes_ragged_rows():
    g = ts.to_grid({"headers": ["a", "b", "c"], "rows": [["x"], ["y", "z"]]})
    assert g == [["a", "b", "c"], ["x", "", ""], ["y", "z", ""]]


def test_score_tables_perfect_none_and_unmatched():
    gold = [{"headers": ["h1", "h2", "h3"], "rows": [["a", "b", "c"], ["d", "e", "f"]]}]
    pred = [{"headers": ["h1", "h2", "h3"], "rows": [["a", "b", "c"], ["d", "e", "f"]]}]
    r = ts.score_tables(gold, pred)
    assert r == {k: 1.0 for k in ts.METRIC_KEYS}
    assert ts.score_tables([], pred) is None                    # no gold tables -> None
    assert ts.score_tables(gold, []) == {k: 0.0 for k in ts.METRIC_KEYS}  # gold, no pred -> 0


def test_score_tables_matches_best_pred_and_averages():
    # Two gold tables; two pred tables in swapped order + one decoy. Content matching must pair them
    # correctly, so both score ~1.0 and the mean is ~1.0 (not confused by order or the decoy).
    t1 = {"headers": ["x"], "rows": [["apple"], ["banana"]]}
    t2 = {"headers": ["y"], "rows": [["cat"], ["dog"]]}
    decoy = {"headers": ["z"], "rows": [["zzz"]]}
    r = ts.score_tables([t1, t2], [t2, decoy, t1])
    assert r["table_teds"] == 1.0 and r["table_grits_con"] == 1.0


def test_greedy_fallback_agrees_with_scipy(monkeypatch):
    con_s, top_s = ts.grits(A, A)
    monkeypatch.setattr(ts, "HAVE_SCIPY", False)
    con_g, top_g = ts.grits(A, A)
    assert (round(con_g, 6), round(top_g, 6)) == (round(con_s, 6), round(top_s, 6)) == (1.0, 1.0)


def test_empty_grids_are_safe():
    assert ts.grits([], A) == (0.0, 0.0)
    assert ts.grits(A, []) == (0.0, 0.0)
    assert ts.teds([], []) == 1.0

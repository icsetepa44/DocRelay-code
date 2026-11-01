from src.eval import table_structure as ts

A = [["h1", "h2", "h3"], ["a", "b", "c"], ["d", "e", "f"]]


def test_identical_grid_scores_one():
    assert ts.teds(A, A) == 1.0


def test_cell_text_diff_lowers_teds():
    B = [["h1", "h2", "h3"], ["a", "b", "c"], ["d", "e", "ZZZ"]]
    assert ts.teds(A, B) < 1.0


def test_missing_row_penalizes_structure():
    C = [["h1", "h2", "h3"], ["a", "b", "c"]]
    assert ts.teds(A, C) < 1.0


def test_transpose_is_not_perfect():
    T = [[A[i][j] for i in range(3)] for j in range(3)]
    assert ts.teds(A, T) < 1.0


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
    assert ts.score_tables([], pred) is None
    assert ts.score_tables(gold, []) == {k: 0.0 for k in ts.METRIC_KEYS}


def test_score_tables_matches_best_pred_and_averages():
    t1 = {"headers": ["x"], "rows": [["apple"], ["banana"]]}
    t2 = {"headers": ["y"], "rows": [["cat"], ["dog"]]}
    decoy = {"headers": ["z"], "rows": [["zzz"]]}
    r = ts.score_tables([t1, t2], [t2, decoy, t1])
    assert r["table_teds"] == 1.0


def test_empty_grids_are_safe():
    assert ts.teds([], []) == 1.0

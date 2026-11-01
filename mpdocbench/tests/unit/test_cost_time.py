"""Per-run cost + time summary (src/eval/cost_time.py). Hermetic: a synthetic ledger.

Proves: cost counts only each doc's kept session (an abandoned attempt is reported apart); $ per 100 pages; cost by
role; failed attempts and calls without a reported cost are counted; wall time sums the sessions; per-page model
time; summarize() writes summary.json from the files and skips a run without a ledger."""

import json

import pytest

from src.eval import cost_time
from src.llm import usage


def _call(session, doc, page, role, cost, status="ok", latency=2.0):
    return {"session": session, "doc": doc, "label": "g", "page": page, "role": role, "status": status,
            "cost_usd": cost, "latency_s": latency, "prompt_tokens": 100, "completion_tokens": 10}


CALLS = [
    _call("s1", "a", 0, "extractor", 0.01),          # doc a, session 1: abandoned (a was reset and redone in s2)
    _call("s2", "a", 0, "extractor", 0.02),
    _call("s2", "a", 0, "super_validator", 0.03),
    _call("s2", "a", 1, "extractor", 0.02),
    _call("s2", "a", None, "final_validator", 0.01),
    _call("s2", "b", 0, "extractor", None, status="error", latency=1.0),   # a failed attempt, then its retry
    _call("s2", "b", 0, "extractor", 0.04),
    _call("s2", "b", 0, "final_decider", None),       # ok but no cost reported
]
PAIRS = [{"session": "s1", "doc": "a", "label": "g", "status": "error", "duration_s": 50.0},
         {"session": "s2", "doc": "a", "label": "g", "status": "done", "duration_s": 100.0},
         {"session": "s2", "doc": "b", "label": "g", "status": "done", "duration_s": 60.0}]
SESSIONS = [{"session": "s1", "wall_s": 30.0}, {"session": "s2", "wall_s": 120.0}]


def test_kept_session_rule_and_totals():
    s = cost_time.summarize_records(CALLS, PAIRS, SESSIONS, {"a": 2, "b": 2})
    assert s["docs_done"] == 2 and s["pages_done"] == 4
    assert s["cost_usd"] == pytest.approx(0.02 + 0.03 + 0.02 + 0.01 + 0.04)
    assert s["abandoned_cost_usd"] == pytest.approx(0.01)
    assert s["cost_per_100_pages_usd"] == pytest.approx(100 * 0.12 / 4)
    assert s["calls"] == 7 and s["failed_attempts"] == 1 and s["calls_missing_cost"] == 1
    assert s["by_role"]["extractor"]["cost_usd"] == pytest.approx(0.08)
    assert s["by_role"]["extractor"]["failed_attempts"] == 1
    assert s["by_role"]["final_validator"]["calls"] == 1
    assert s["by_role"]["extractor"]["latency_s"] == pytest.approx(2.0 + 2.0 + 1.0 + 2.0)   # kept attempts only
    assert s["by_role"]["extractor"]["prompt_tokens"] == 400 and s["by_role"]["extractor"]["completion_tokens"] == 40
    assert s["wall_time_s"] == pytest.approx(150.0) and s["sessions"] == 2
    assert s["doc_time_sum_s"] == pytest.approx(160.0)          # kept pair attempts only
    assert s["wall_s_per_page"] == pytest.approx(150.0 / 4)
    # per-page model time: a/0 = 2+2, a/1 = 2, b/0 = 1+2+2 -> median 4
    assert s["model_s_per_page_median"] == pytest.approx(4.0)


def test_unfinished_doc_costs_nothing_in_the_total():
    pairs = [p for p in PAIRS if p["doc"] != "b"]
    s = cost_time.summarize_records(CALLS, pairs, SESSIONS, {"a": 2, "b": 2})
    assert s["docs_done"] == 1 and s["pages_done"] == 2
    assert s["abandoned_cost_usd"] == pytest.approx(0.01 + 0.04)  # b's calls belong to no kept attempt


def test_summarize_reads_the_files(tmp_path):
    assert cost_time.summarize(tmp_path) is None                 # no ledger -> nothing written
    d = tmp_path / usage.COST_TIME_DIR
    d.mkdir()
    for name, rows in ((usage.CALLS_FILE, CALLS), (usage.PAIRS_FILE, PAIRS), (usage.SESSIONS_FILE, SESSIONS)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"torn', encoding="utf-8")
    for doc in ("a", "b"):
        (tmp_path / doc / "g").mkdir(parents=True)
        (tmp_path / doc / "g" / "run_summary.json").write_text(json.dumps({"pages": 2}), encoding="utf-8")
    s = cost_time.summarize(tmp_path)
    assert s["run"] == tmp_path.name and s["pages_done"] == 4
    assert json.loads((d / "summary.json").read_text(encoding="utf-8"))["cost_usd"] == s["cost_usd"]

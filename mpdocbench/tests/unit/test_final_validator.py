from src.core import config
from src.stages.final_validator import _coverage_gate


def test_coverage_gate_trips_over_threshold(monkeypatch):
    monkeypatch.setattr(config, "MAX_EMPTY_PAGE_RATIO", 0.15)
    summaries = [{"empty_nontrivial": True}] * 3 + [{"empty_nontrivial": False}] * 5  # 3/8 = 0.375
    empty, total, ratio, tripped = _coverage_gate(summaries)
    assert empty == 3 and total == 8 and tripped is True


def test_coverage_gate_ok_under_threshold(monkeypatch):
    monkeypatch.setattr(config, "MAX_EMPTY_PAGE_RATIO", 0.15)
    summaries = [{"empty_nontrivial": False}] * 8
    _, _, _, tripped = _coverage_gate(summaries)
    assert tripped is False


def test_coverage_gate_empty_input():
    empty, total, ratio, tripped = _coverage_gate([])
    assert total == 0 and tripped is False

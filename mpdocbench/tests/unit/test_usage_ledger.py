"""Cost/time ledger (src/llm/usage.py + the client hook). Hermetic: the OpenAI client is faked, no API.

Proves: one tagged line per HTTP attempt with the provider-reported cost and tokens; a retried call logs both
attempts; usage accounting is requested without dropping the provider pin; nothing is written (and nothing breaks)
when no ledger is active; worker threads keep their own tags."""

import json
import threading
from types import SimpleNamespace

import openai
import pytest

from src.llm import client, usage

CFG = client.LLMRequestConfig(backend="openrouter", model="google/gemma-4-31b-it", max_retries=2, retry_backoff=0.0,
                              provider={"quantizations": ["bf16"], "allow_fallbacks": True})


def _response(text='{"ok": 1}', cost=0.0012, extra_cost=None):
    usage_obj = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, cost=cost,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                                prompt_tokens_details=SimpleNamespace(cached_tokens=10),
                                model_extra={"cost": extra_cost} if extra_cost is not None else {})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage_obj,
                           model="google/gemma-4-31b-it", id="gen-1", provider="DeepInfra")


class FakeOpenAI:
    """Stands in for openai.OpenAI: answers from a script of responses / exceptions and keeps every request."""
    script, requests = [], []

    def __init__(self, **kw):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        FakeOpenAI.requests.append(kw)
        nxt = FakeOpenAI.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    FakeOpenAI.script, FakeOpenAI.requests = [], []
    return FakeOpenAI


def _rows(run_dir):
    path = run_dir / "_cost_time" / "llm_calls.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _call():
    return client.chat_completion(system_prompt="s", user_prompt="u", config=CFG)


def test_one_tagged_line_per_attempt_with_cost(fake, tmp_path):
    fake.script = [_response()]
    with usage.ledger(tmp_path, "sess-1"), usage.tags(doc="d1", label="gemma"):
        usage.set_tags(page=3)
        with usage.tags(role="extractor", repair=0):
            assert _call() == '{"ok": 1}'
    (row,) = _rows(tmp_path)
    assert row["session"] == "sess-1" and row["doc"] == "d1" and row["label"] == "gemma"
    assert row["page"] == 3 and row["role"] == "extractor" and row["repair"] == 0
    assert row["status"] == "ok" and row["model"] == "google/gemma-4-31b-it" and row["provider"] == "DeepInfra"
    assert row["cost_usd"] == pytest.approx(0.0012)
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]) == (1000, 200, 10)
    assert row["latency_s"] >= 0


def test_usage_accounting_requested_and_provider_pin_kept(fake, tmp_path):
    fake.script = [_response()]
    _call()
    extra = fake.requests[0]["extra_body"]
    assert extra["usage"] == {"include": True}
    assert extra["provider"] == {"quantizations": ["bf16"], "allow_fallbacks": True}


def test_retried_call_logs_both_attempts(fake, tmp_path):
    fake.script = [RuntimeError("429 rate limit"), _response(cost=0.002)]
    with usage.ledger(tmp_path, "s"), usage.tags(role="super_validator"):
        _call()
    rows = _rows(tmp_path)
    assert [r["status"] for r in rows] == ["error", "ok"]
    assert "429" in rows[0]["error"] and rows[0].get("cost_usd") is None
    assert rows[1]["cost_usd"] == pytest.approx(0.002)
    assert {r["role"] for r in rows} == {"super_validator"}


def test_cost_from_pydantic_extra(fake, tmp_path):
    fake.script = [_response(cost=None, extra_cost=0.5)]
    with usage.ledger(tmp_path, "s"):
        _call()
    assert _rows(tmp_path)[0]["cost_usd"] == pytest.approx(0.5)


def test_no_ledger_no_write(fake, tmp_path):
    fake.script = [_response()]
    assert _call() == '{"ok": 1}'
    assert not (tmp_path / "_cost_time").exists()
    usage.record(model="x", status="ok")  # a stray record without a ledger is a no-op
    usage.append(usage.PAIRS_FILE, {"doc": "x"})


def test_worker_threads_keep_their_own_tags(tmp_path):
    def work(doc):
        with usage.tags(doc=doc):
            for page in range(3):
                usage.set_tags(page=page)
                usage.record(model="m", status="ok")

    with usage.ledger(tmp_path, "s"):
        threads = [threading.Thread(target=work, args=(f"doc{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    rows = _rows(tmp_path)
    assert len(rows) == 12
    assert sorted((r["doc"], r["page"]) for r in rows) == sorted((f"doc{i}", p) for i in range(4) for p in range(3))


def test_tags_restore_after_block():
    with usage.tags(doc="a"):
        usage.set_tags(page=1)
        with usage.tags(role="x"):
            assert usage._TAGS.get() == {"doc": "a", "page": 1, "role": "x"}
        assert usage._TAGS.get() == {"doc": "a", "page": 1}
    assert "doc" not in usage._TAGS.get()

"""
Cost + time ledger for LLM calls. Pure logging: it never raises and never changes a call.

One JSON line per HTTP attempt goes to ``runs/<run>/_cost_time/llm_calls.jsonl``. Each line says who called
(session, doc, model label, page, role, repair attempt), what it cost (the provider-reported ``usage.cost`` and the
token counts OpenRouter returns) and how long it took. run.py adds per-pair and per-session timing lines next to it
(``pairs.jsonl``, ``sessions.jsonl``); ``src/eval/cost_time.py`` sums them into ``summary.json``.

    with usage.ledger(run_dir, session):           # run.py, once per run session (shared by all worker threads)
        with usage.tags(doc=doc, label=label):     # run.py, per (doc, model) pair, inside its worker thread
            usage.set_tags(page=3)                 # orchestrator, per page
            with usage.tags(role="extractor"):     # structured_output / validator, per logical call
                ...                                # client.py calls usage.record(...) after every attempt

The ledger is module-level (one run at a time; worker threads do not inherit context variables), the tags are a
ContextVar, so each worker thread keeps its own doc/page/role.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

COST_TIME_DIR = "_cost_time"   # under the run dir
CALLS_FILE = "llm_calls.jsonl"
PAIRS_FILE = "pairs.jsonl"
SESSIONS_FILE = "sessions.jsonl"

_TAGS: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("llm_usage_tags", default={})
_LOCK = threading.Lock()
_ACTIVE: Optional[Dict[str, Any]] = None   # {"dir": Path, "session": str}


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


@contextlib.contextmanager
def ledger(run_dir: Path, session: str) -> Iterator[Path]:
    """Activate the ledger of one run for the duration of a session; yields its ``_cost_time`` dir."""
    global _ACTIVE
    out = Path(run_dir) / COST_TIME_DIR
    out.mkdir(parents=True, exist_ok=True)
    prev, _ACTIVE = _ACTIVE, {"dir": out, "session": session}
    try:
        yield out
    finally:
        _ACTIVE = prev


@contextlib.contextmanager
def tags(**kw: Any) -> Iterator[None]:
    """Merge tags into this thread's context for the block (restored on exit)."""
    token = _TAGS.set({**_TAGS.get(), **kw})
    try:
        yield
    finally:
        _TAGS.reset(token)


def set_tags(**kw: Any) -> None:
    """Merge tags until the enclosing ``tags()`` block exits (e.g. the page number inside a doc)."""
    _TAGS.set({**_TAGS.get(), **kw})


def append(name: str, row: Dict[str, Any]) -> None:
    """Append one JSON line to ``<active ledger dir>/<name>``; a no-op when no ledger is active."""
    led = _ACTIVE
    if led is None:
        return
    try:
        line = json.dumps({"session": led["session"], **row}, ensure_ascii=False, default=str)
        with _LOCK, open(led["dir"] / name, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 — logging must never break a run
        pass


def record(**fields: Any) -> None:
    """One LLM attempt: the current tags + ``fields`` (model, latency_s, status, tokens, cost_usd, ...)."""
    if _ACTIVE is None:
        return
    append(CALLS_FILE, {"ts": now_iso(), **_TAGS.get(), **fields})


def _get(obj: Any, name: str) -> Any:
    """Attribute or dict key or pydantic extra (``model_extra``), else None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    val = getattr(obj, name, None)
    if val is None:
        val = (getattr(obj, "model_extra", None) or {}).get(name)
    return val


def from_response(response: Any) -> Dict[str, Any]:
    """Usage fields of an OpenAI-compatible (OpenRouter) chat-completion response. Missing fields -> None."""
    try:
        u = _get(response, "usage")
        cost = _get(u, "cost")
        return {
            "served_model": _get(response, "model"),
            "provider": _get(response, "provider"),
            "generation_id": _get(response, "id"),
            "prompt_tokens": _get(u, "prompt_tokens"),
            "completion_tokens": _get(u, "completion_tokens"),
            "reasoning_tokens": _get(_get(u, "completion_tokens_details"), "reasoning_tokens"),
            "cached_tokens": _get(_get(u, "prompt_tokens_details"), "cached_tokens"),
            "cost_usd": float(cost) if cost is not None else None,
        }
    except Exception:  # noqa: BLE001
        return {"cost_usd": None}

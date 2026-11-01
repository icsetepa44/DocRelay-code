from __future__ import annotations

import contextlib
import contextvars
import datetime
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

COST_TIME_DIR = "_cost_time"
CALLS_FILE = "llm_calls.jsonl"
PAIRS_FILE = "pairs.jsonl"
SESSIONS_FILE = "sessions.jsonl"

_TAGS: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("llm_usage_tags", default={})
_LOCK = threading.Lock()
_ACTIVE: Optional[Dict[str, Any]] = None


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


@contextlib.contextmanager
def ledger(run_dir: Path, session: str) -> Iterator[Path]:
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
    token = _TAGS.set({**_TAGS.get(), **kw})
    try:
        yield
    finally:
        _TAGS.reset(token)


def set_tags(**kw: Any) -> None:
    _TAGS.set({**_TAGS.get(), **kw})


def append(name: str, row: Dict[str, Any]) -> None:
    led = _ACTIVE
    if led is None:
        return
    try:
        line = json.dumps({"session": led["session"], **row}, ensure_ascii=False, default=str)
        with _LOCK, open(led["dir"] / name, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def record(**fields: Any) -> None:
    if _ACTIVE is None:
        return
    append(CALLS_FILE, {"ts": now_iso(), **_TAGS.get(), **fields})


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    val = getattr(obj, name, None)
    if val is None:
        val = (getattr(obj, "model_extra", None) or {}).get(name)
    return val


def from_response(response: Any) -> Dict[str, Any]:
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
    except Exception:
        return {"cost_usd": None}

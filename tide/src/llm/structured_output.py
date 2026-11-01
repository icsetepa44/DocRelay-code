"""
Structured-output layer: build per-role request configs, call the LLM, extract the
JSON object, validate it against a JSON Schema, and (on failure) do one corrective
re-ask. This compensates for the client only supporting `json_object` (not provider
`json_schema`).

    obj = request_json(
        role="extractor",
        system_prompt=...,
        user_prompt=...,
        schema_name="page_extraction",
        images=[pil_page_image],
    )
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator

from src.core import config
from src.core.exceptions import StructuredOutputError
from src.llm.client import LLMRequestConfig, chat_completion


# ------------------------------------------------------------------
# Request-config factory (globals -> LLMRequestConfig)
# ------------------------------------------------------------------

def build_request_config(
    role: str,
    response_format: str = "json",
    model_override: Optional[Dict[str, Any]] = None,
) -> LLMRequestConfig:
    m = model_override or config.model_for(role)
    return LLMRequestConfig(
        backend=m["backend"],
        model=m["model"],
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_new_tokens=config.MAX_NEW_TOKENS,
        response_format=response_format,
        timeout=config.TIMEOUT_SECONDS,
        max_retries=config.MAX_RETRIES,
        retry_backoff=config.RETRY_BACKOFF_SECONDS,
        provider=config.provider_for(m),
    )


# ------------------------------------------------------------------
# JSON parsing + schema validation
# ------------------------------------------------------------------

def _salvage_json(text: str) -> Optional[Dict[str, Any]]:
    """Last-resort STRUCTURE-ONLY repair of a nearly-valid JSON object (C1).

    Two safe repairs, neither of which ever alters text inside a string (so model
    content is never invented or changed):
      * a TRUNCATED object (the model was cut off mid-output, e.g. hit the token cap)
        is trimmed back to the last completed nested container and its still-open
        brackets are closed — this recovers the blocks that DID finish instead of
        losing the whole page;
      * a trailing comma right before a `}`/`]` is dropped.
    Returns the parsed dict, or None if it still cannot be parsed.
    """
    start = text.find("{")
    if start < 0:
        return None
    out: List[str] = []
    stack: List[str] = []          # expected closing brackets, outermost first
    in_str = False
    escape = False
    last_complete = None            # (len(out), open-stack) just after a nested container closed
    for ch in text[start:]:
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
        elif ch in "}]":
            k = len(out) - 1                      # structure-aware trailing-comma removal
            while k >= 0 and out[k] in " \t\r\n":
                k -= 1
            if k >= 0 and out[k] == ",":
                del out[k]
            out.append(ch)
            if stack:
                stack.pop()
            if not stack:                          # completed the whole top-level object
                try:
                    return json.loads("".join(out))
                except json.JSONDecodeError:
                    return None
            last_complete = (len(out), list(stack))
        else:
            out.append(ch)
    if last_complete is None:                      # truncated with no completed container
        return None
    cut_len, open_stack = last_complete
    repaired = out[:cut_len]
    k = len(repaired) - 1
    while k >= 0 and repaired[k] in " \t\r\n":
        k -= 1
    if k >= 0 and repaired[k] == ",":
        del repaired[k]
    candidate = "".join(repaired) + "".join(reversed(open_stack))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def extract_json_object(text: str) -> Dict[str, Any]:
    """Direct JSON parse; else the largest {...} object; else a structure-only salvage (C1)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    salvaged = _salvage_json(text)          # C1: recover truncated/trailing-comma output
    if salvaged is not None:
        return salvaged
    if not match:
        raise StructuredOutputError("No JSON object found in model output.")
    raise StructuredOutputError("Malformed JSON in model output (unrecoverable after salvage).")


@lru_cache(maxsize=32)
def load_schema(schema_name: str) -> str:
    """Load schemas/<name>.schema.json (returns JSON text; cached)."""
    path = config.SCHEMAS_DIR / f"{schema_name}.schema.json"
    return path.read_text(encoding="utf-8")


def _validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads(load_schema(schema_name)))


def schema_errors(obj: Any, schema_name: str) -> List[str]:
    errors = sorted(_validator(schema_name).iter_errors(obj), key=lambda e: list(e.path))
    out = []
    for e in errors:
        loc = "/".join(str(p) for p in e.path) or "<root>"
        out.append(f"{loc}: {e.message}")
    return out


def validate(obj: Any, schema_name: str) -> None:
    errs = schema_errors(obj, schema_name)
    if errs:
        raise StructuredOutputError(
            f"Output failed schema '{schema_name}':\n- " + "\n- ".join(errs[:8])
        )


def _set_at_path(root: Any, path: Any, value: Any) -> None:
    """Set `root` at a jsonschema error path (deque of keys/indices) to `value` (best-effort)."""
    keys = list(path)
    if not keys:
        return
    cur = root
    try:
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    except (TypeError, KeyError, IndexError):
        pass


def _coerce_null_strings(obj: Any, schema_name: str) -> bool:
    """Heal a `null` where the schema requires a non-null string by setting it to "".

    Uses the validator's own type errors to locate exactly those paths, so valid content is never
    altered. Returns True if anything changed. Robustness net for models that echo `null` for a
    field they had no input for (the mixed-OCR ``"html": null`` class of failure — FAILURE_ANALYSIS_V8).
    """
    changed = False
    for err in _validator(schema_name).iter_errors(obj):
        if err.validator == "type" and err.instance is None:
            expected = err.validator_value
            types = expected if isinstance(expected, list) else [expected]
            if "string" in types and "null" not in types:
                _set_at_path(obj, err.absolute_path, "")
                changed = True
    return changed


# ------------------------------------------------------------------
# The main helper: call -> parse -> validate -> (corrective re-ask)
# ------------------------------------------------------------------

def request_json(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    images: Optional[List[Any]] = None,
    max_repair: int = 2,  # C1: 3 total attempts (was 2) — one extra corrective re-ask
    model_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = build_request_config(role, model_override=model_override)
    prompt = user_prompt
    last_error: Optional[str] = None

    for attempt in range(max_repair + 1):
        if attempt > 0 and last_error:
            prompt = (
                user_prompt
                + "\n\n---\nYour previous response was INVALID:\n"
                + last_error
                + "\n\nReturn ONLY one valid JSON object that satisfies the required "
                  "schema. No markdown, no commentary, no text before or after the JSON."
            )
        raw = chat_completion(
            system_prompt=system_prompt,
            user_prompt=prompt,
            config=cfg,
            images=images,
        )
        try:
            obj = extract_json_object(raw)
            try:
                validate(obj, schema_name)
            except StructuredOutputError:
                # Robustness: heal `null`-where-a-string-is-required before giving up on this
                # attempt (belt-and-suspenders with the nullable-field schemas).
                if _coerce_null_strings(obj, schema_name):
                    validate(obj, schema_name)  # re-validate; may still raise -> corrective re-ask
                else:
                    raise
            return obj
        except StructuredOutputError as exc:
            last_error = str(exc)

    raise StructuredOutputError(
        f"Role '{role}' produced invalid '{schema_name}' output after "
        f"{max_repair + 1} attempt(s). Last error:\n{last_error}"
    )

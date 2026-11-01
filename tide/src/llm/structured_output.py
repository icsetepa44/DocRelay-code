from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator

from src.core import config
from src.core.exceptions import StructuredOutputError
from src.llm.client import LLMRequestConfig, chat_completion


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


def _salvage_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start < 0:
        return None
    out: List[str] = []
    stack: List[str] = []
    in_str = False
    escape = False
    last_complete = None
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
            k = len(out) - 1
            while k >= 0 and out[k] in " \t\r\n":
                k -= 1
            if k >= 0 and out[k] == ",":
                del out[k]
            out.append(ch)
            if stack:
                stack.pop()
            if not stack:
                try:
                    return json.loads("".join(out))
                except json.JSONDecodeError:
                    return None
            last_complete = (len(out), list(stack))
        else:
            out.append(ch)
    if last_complete is None:
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
    salvaged = _salvage_json(text)
    if salvaged is not None:
        return salvaged
    if not match:
        raise StructuredOutputError("No JSON object found in model output.")
    raise StructuredOutputError("Malformed JSON in model output (unrecoverable after salvage).")


@lru_cache(maxsize=32)
def load_schema(schema_name: str) -> str:
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
    changed = False
    for err in _validator(schema_name).iter_errors(obj):
        if err.validator == "type" and err.instance is None:
            expected = err.validator_value
            types = expected if isinstance(expected, list) else [expected]
            if "string" in types and "null" not in types:
                _set_at_path(obj, err.absolute_path, "")
                changed = True
    return changed


def request_json(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    images: Optional[List[Any]] = None,
    max_repair: int = 2,
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
                if _coerce_null_strings(obj, schema_name):
                    validate(obj, schema_name)
                else:
                    raise
            return obj
        except StructuredOutputError as exc:
            last_error = str(exc)

    raise StructuredOutputError(
        f"Role '{role}' produced invalid '{schema_name}' output after "
        f"{max_repair + 1} attempt(s). Last error:\n{last_error}"
    )

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

from src.core import config

_TOKEN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    path = config.PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _effective_name(name: str, domain: Optional[str]) -> str:
    if domain:
        override = f"{name}__{domain}"
        if (config.PROMPTS_DIR / f"{override}.md").exists():
            return override
    return name


def render_prompt(name: str, *, domain: Optional[str] = None, **kwargs: Any) -> str:
    kwargs.setdefault("domain_profile",
                      config.domain_profile(domain)["guidance"] if domain else "")
    text = load_prompt(_effective_name(name, domain))

    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        return str(kwargs[key]) if key in kwargs else match.group(0)

    return _TOKEN.sub(repl, text)

"""
Prompt loader — reads prompts/<name>.md and substitutes {{placeholder}} tokens.

Prompts are stored as files (spec §9-18), not inline. Placeholders use a DOUBLE-brace
syntax `{{name}}` so that single braces `{ }` in embedded JSON examples are left
untouched. Unknown `{{tokens}}` are left intact rather than raising.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

from src.core import config

_TOKEN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """Load raw prompt text from prompts/<name>.md (cached)."""
    path = config.PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _effective_name(name: str, domain: Optional[str]) -> str:
    """prefer a per-domain override file ``<name>__<domain>.md`` when it exists, else base."""
    if domain:
        override = f"{name}__{domain}"
        if (config.PROMPTS_DIR / f"{override}.md").exists():
            return override
    return name


def render_prompt(name: str, *, domain: Optional[str] = None, **kwargs: Any) -> str:
    """Load prompt `name` and replace {{tokens}} with values; unknown tokens stay as-is.

    when `domain` is given, a per-role override `prompts/<name>__<domain>.md` is used if
    present (base otherwise). The `{{domain_profile}}` token is ALWAYS filled — from the domain's
    profile guidance, or `""` when there is no domain — so a prompt carrying that token never
    leaves a literal `{{domain_profile}}` behind.
    """
    kwargs.setdefault("domain_profile",
                      config.domain_profile(domain)["guidance"] if domain else "")
    text = load_prompt(_effective_name(name, domain))

    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        return str(kwargs[key]) if key in kwargs else match.group(0)

    return _TOKEN.sub(repl, text)

"""
Prompt loader — reads prompts/<name>.md and substitutes {{placeholder}} tokens.

Prompts are stored as files (spec §9-18), not inline. Placeholders use a DOUBLE-brace
syntax `{{name}}` so that single braces `{ }` in embedded JSON examples are left
untouched. Unknown `{{tokens}}` are left intact rather than raising.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.core import config

_TOKEN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """Load raw prompt text from prompts/<name>.md (cached)."""
    path = config.PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: Any) -> str:
    """Load prompt `name` and replace {{tokens}} with values; unknown tokens stay as-is."""
    text = load_prompt(name)

    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        return str(kwargs[key]) if key in kwargs else match.group(0)

    return _TOKEN.sub(repl, text)

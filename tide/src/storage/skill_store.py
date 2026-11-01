from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core import config
from src.core.exceptions import SkillRegistryError


def registry_path() -> Path:
    return config.SKILLS_DIR / "registry.json"


def read_registry() -> Dict[str, Any]:
    path = registry_path()
    if not path.exists():
        raise SkillRegistryError(f"Skill registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_record_by_relpath(relpath: str) -> Dict[str, Any]:
    path = config.SKILLS_DIR / relpath
    if not path.exists():
        raise SkillRegistryError(f"Skill record not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_registry_entries() -> List[Dict[str, Any]]:
    return list(read_registry().get("skills", []))

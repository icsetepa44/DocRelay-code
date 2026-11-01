from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.contracts.skill import SkillRecord
from src.core.exceptions import SkillRegistryError
from src.storage import skill_store


class SkillRegistry:
    def __init__(self) -> None:
        self._by_version: Dict[Tuple[str, int], SkillRecord] = {}
        self._latest: Dict[str, SkillRecord] = {}
        self._status: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for entry in skill_store.list_registry_entries():
            skill_id = str(entry["skill_id"])
            relpath = str(entry["record"])
            record = SkillRecord.from_dict(skill_store.read_record_by_relpath(relpath))
            self._by_version[(record.skill_id, record.version)] = record
            self._latest[skill_id] = record
            self._status[skill_id] = str(entry.get("status", record.status))


    def get(self, skill_id: str, version: Optional[int] = None) -> SkillRecord:
        if version is None:
            if skill_id not in self._latest:
                raise SkillRegistryError(f"Unknown skill_id: {skill_id}")
            return self._latest[skill_id]
        key = (skill_id, int(version))
        if key not in self._by_version:
            if skill_id in self._latest:
                return self._latest[skill_id]
            raise SkillRegistryError(f"Unknown skill {skill_id} v{version}")
        return self._by_version[key]

    def by_status(self, status: str) -> List[SkillRecord]:
        return [r for sid, r in self._latest.items() if self._status.get(sid) == status]

    def resolve(self, selected: Iterable[Any]) -> List[SkillRecord]:
        out: List[SkillRecord] = []
        seen = set()
        for item in selected:
            if isinstance(item, dict):
                sid = str(item.get("skill_id", ""))
                ver = item.get("version")
            else:
                sid = str(getattr(item, "skill_id", ""))
                ver = getattr(item, "version", None)
            if not sid or sid in seen:
                continue
            try:
                rec = self.get(sid, int(ver) if ver is not None else None)
            except SkillRegistryError:
                continue
            seen.add(sid)
            out.append(rec)
        return out


_REGISTRY: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SkillRegistry()
    return _REGISTRY

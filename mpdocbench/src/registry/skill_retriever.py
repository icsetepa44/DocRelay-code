from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

from src.contracts.skill import SkillRecord
from src.registry.skill_registry import SkillRegistry, get_registry
from src.registry import embedding_scorer
from src.core import config

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def _score(query_tokens: Counter, doc_tokens: Counter) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    common = set(query_tokens) & set(doc_tokens)
    dot = sum(query_tokens[t] * doc_tokens[t] for t in common)
    qn = math.sqrt(sum(v * v for v in query_tokens.values()))
    dn = math.sqrt(sum(v * v for v in doc_tokens.values()))
    return dot / (qn * dn) if qn and dn else 0.0


def retrieve_skills(
    query: str,
    statuses: List[str],
    top_k: int,
    registry: Optional[SkillRegistry] = None,
    pin_skill_ids: Optional[List[str]] = None,
    weight_skill_ids: Optional[Dict[str, float]] = None,
) -> List[SkillRecord]:
    registry = registry or get_registry()
    candidates: List[SkillRecord] = []
    for status in statuses:
        candidates.extend(registry.by_status(status))

    seen = set()
    unique: List[SkillRecord] = []
    for rec in candidates:
        if rec.skill_id in seen:
            continue
        seen.add(rec.skill_id)
        unique.append(rec)

    if config.RETRIEVAL_SCORER == "embedding":
        scores = embedding_scorer.score(query, [rec.retrieval_text() for rec in unique])
        scored = list(zip(unique, scores))
    else:
        q = Counter(_tokens(query))
        scored = [(rec, _score(q, Counter(_tokens(rec.retrieval_text())))) for rec in unique]
    if weight_skill_ids:
        scored = [(rec, score + float(weight_skill_ids.get(rec.skill_id, 0.0)))
                  for rec, score in scored]
    scored.sort(key=lambda x: (-x[1], x[0].skill_id))
    result = [rec for rec, _ in scored[:top_k]]

    if pin_skill_ids:
        have = {rec.skill_id for rec in result}
        by_id = {rec.skill_id: rec for rec in unique}
        for sid in pin_skill_ids:
            if sid not in have and sid in by_id:
                result.append(by_id[sid])
                have.add(sid)
    return result

"""
Skill Retriever (spec §8.2) — a swappable similarity index over each skill's
description + applicability. Operational `instructions` are intentionally NOT indexed.

The scorer is config-selected (skill_retrieval.method): "keyword" (stdlib TF-cosine) or
"embedding" (sentence-transformers); run.py currently sets "embedding". With main_top_k=3
and 6 main skills, retrieval keeps only the 3 highest-scoring candidates per query, and the
Skill Router (LLM) makes the final selection from those.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional

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
    # cosine similarity over raw term frequencies
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
) -> List[SkillRecord]:
    """Return the top_k most similar skills, plus any pinned skills.

    ``pin_skill_ids`` force-includes named skills (appended after the top_k when not already
    present) so a caller can guarantee availability regardless of the similarity score — used
    to keep the ``table`` skill on offer for a continuing table (T7). Pins are looked up from
    the candidate set, so a pin that names no existing skill is silently ignored. The Skill
    Router still makes the final selection from whatever is returned.
    """
    registry = registry or get_registry()
    candidates: List[SkillRecord] = []
    for status in statuses:
        candidates.extend(registry.by_status(status))

    # de-dup while preserving order
    seen = set()
    unique: List[SkillRecord] = []
    for rec in candidates:
        if rec.skill_id in seen:
            continue
        seen.add(rec.skill_id)
        unique.append(rec)

    # Scorer is selected live from config so a runner can override it at runtime.
    if config.RETRIEVAL_SCORER == "embedding":
        scores = embedding_scorer.score(query, [rec.retrieval_text() for rec in unique])
        scored = list(zip(unique, scores))
    else:
        q = Counter(_tokens(query))
        scored = [(rec, _score(q, Counter(_tokens(rec.retrieval_text())))) for rec in unique]
    # stable sort: score desc, then skill_id for determinism
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

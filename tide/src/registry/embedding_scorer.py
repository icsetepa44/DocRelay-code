from __future__ import annotations

import threading
from typing import Dict, List

from src.core import config

_lock = threading.RLock()
_model = None
_doc_cache: Dict[str, "object"] = {}


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _embed(texts: List[str]):
    with _lock:
        model = _get_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def score(query: str, docs: List[str]) -> List[float]:
    if not docs:
        return []
    import numpy as np

    q_vec = _embed([config.EMBEDDING_QUERY_PREFIX + (query or "")])[0]

    missing = [d for d in docs if d not in _doc_cache]
    if missing:
        vecs = _embed([config.EMBEDDING_DOC_PREFIX + d for d in missing])
        for d, v in zip(missing, vecs):
            _doc_cache[d] = v

    doc_vecs = np.array([_doc_cache[d] for d in docs])
    sims = doc_vecs @ q_vec
    return [float(s) for s in sims]


def warm_up() -> int:
    model = _get_model()
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(get_dim())

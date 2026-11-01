"""
Embedding-based scorer for skill retrieval — the optional "embedding" backend selected by
`config.RETRIEVAL_SCORER == "embedding"` (config key `skill_retrieval.method`).

It ranks each skill's `retrieval_text()` against the retrieval query by cosine similarity of
sentence-transformers embeddings (default `intfloat/multilingual-e5-small`, Bengali+English).
Kept in its own module so the default keyword path in `skill_retriever` never imports torch.

The model and the (few, stable) skill embeddings are loaded/computed lazily once and cached.
A reentrant lock serializes model load AND `encode()` calls, because the benchmark runner scores
from several extractor-model threads concurrently and a shared torch module is not guaranteed
safe under concurrent forward passes. The per-page workload (1 query + <=6 short docs) is tiny,
so serializing is effectively free.
"""

from __future__ import annotations

import threading
from typing import Dict, List

from src.core import config

_lock = threading.RLock()
_model = None                        # SentenceTransformer, loaded lazily
_doc_cache: Dict[str, "object"] = {}  # doc text -> normalized embedding (np.ndarray)


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # Imported lazily so the keyword path never pays the torch import cost.
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _embed(texts: List[str]):
    """Encode texts to L2-normalized vectors (so a dot product equals cosine similarity)."""
    with _lock:
        model = _get_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def score(query: str, docs: List[str]) -> List[float]:
    """Cosine similarity of `query` against each doc; returns one float per doc, in order."""
    if not docs:
        return []
    import numpy as np

    q_vec = _embed([config.EMBEDDING_QUERY_PREFIX + (query or "")])[0]

    # Cache doc embeddings by exact text — skills are few and their retrieval_text is stable.
    missing = [d for d in docs if d not in _doc_cache]
    if missing:
        vecs = _embed([config.EMBEDDING_DOC_PREFIX + d for d in missing])
        for d, v in zip(missing, vecs):
            _doc_cache[d] = v

    doc_vecs = np.array([_doc_cache[d] for d in docs])
    sims = doc_vecs @ q_vec               # all vectors normalized -> dot == cosine
    return [float(s) for s in sims]


def warm_up() -> int:
    """Force the model to load (triggering any one-time download) and return its embedding dim.

    Used by the benchmark runner to fail fast / prime the offline cache before the run starts.
    """
    model = _get_model()
    # method was renamed across sentence-transformers versions; support both.
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(get_dim())

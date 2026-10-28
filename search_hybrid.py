"""
Hybrid retrieval helpers for the question answering service.

Pure, dependency-free functions shared by web.py and the test suite: query
normalization (deterministic embedding/cache keys), Reciprocal Rank Fusion for
combining the dense (kNN) and lexical (BM25) retrieval legs, and deduplication
of fused pools.
"""

import hashlib
import re
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

_T = TypeVar("_T")


def normalize_query(text: str) -> str:
    """Normalize a user question before embedding or retrieval.

    Applies NFKC unicode normalization and collapses all whitespace runs to a
    single space, so semantically identical questions (differing only in
    spacing, line breaks, or unicode compatibility forms) produce identical
    embedding inputs and cache keys.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def query_hash(text: str) -> str:
    """Stable sha256 hex digest of the normalized question."""
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    limit: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Fuse one or more ranked document-id lists with Reciprocal Rank Fusion.

    Each ranking contributes ``1 / (k + rank + 1)`` per document, where rank
    is the 0-based position within that ranking. Documents appearing in
    several rankings accumulate the sum of their contributions, which makes
    the fused order robust against a single noisy leg. Ties are broken by
    first-seen order across the rankings, keeping the result deterministic.

    Args:
        rankings: Ranked lists of document identifiers, best first.
        k: RRF constant (60 in the original RRF paper).
        limit: Maximum number of documents to return.

    Returns:
        List of ``(doc_id, fused_score)`` tuples sorted by fused score,
        best first.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking):
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda item: -item[1])
    if limit is not None:
        fused = fused[:limit]
    return fused


def dedupe(items: Iterable[_T], key: Callable[[_T], str]) -> List[_T]:
    """Keep the first item per key value, preserving input order.

    The corpus contains the same decision indexed multiple times (different
    resource ids, identical description text); collapsing them keeps a single
    copy of each underlying document in the pool, preferring the best-ranked
    copy.
    """
    seen = set()
    unique: List[_T] = []
    for item in items:
        value = key(item)
        if value in seen:
            continue
        seen.add(value)
        unique.append(item)
    return unique

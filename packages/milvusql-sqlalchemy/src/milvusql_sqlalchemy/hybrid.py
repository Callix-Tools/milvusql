"""``hybrid_search()`` -- the one genuinely new Core-level construct
this dialect adds (D3).

Why a function, not a ``Comparator`` method: hybrid search takes two or
more weighted arms, and there is no single column to hang a method
call on the way ``l2_distance``/``cosine_distance`` (``types.py``) hang
off one. Why ``.order_by(hybrid_search(...))``, not ``.where(...)``:
in MilvusQL's own grammar (Track A), ``HYBRID SEARCH ... RERANK ...``
sits exactly where ``ORDER BY`` would in a plain vector search -- it
*is* the ranking criterion, just for two-or-more vectors instead of
one. ``compiler.py``'s ``order_by_clause`` override renders it there,
without the literal ``ORDER BY`` keyword MilvusQL doesn't use for this
clause.
"""

from __future__ import annotations

import typing as t

from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.type_api import NULLTYPE


class SearchArm(ColumnElement):
    """One weighted arm: ``embedding <=> :q WEIGHT 0.7``. ``distance``
    is whatever a ``VECTOR``/``SPARSEVEC`` comparator method already
    produces (``types.py``) -- this only attaches the weight."""

    __visit_name__ = "milvus_search_arm"
    inherit_cache = True

    def __init__(self, distance: ColumnElement, weight: float) -> None:
        self.distance = distance
        self.weight = weight
        self.type = NULLTYPE


class HybridSearchClause(ColumnElement):
    """``HYBRID SEARCH (<arm>, ...) RERANK <strategy>(<params>)``."""

    __visit_name__ = "milvus_hybrid_search"
    inherit_cache = True

    def __init__(
        self,
        arms: t.Sequence[SearchArm],
        rerank: str,
        rerank_params: dict[str, t.Any],
    ) -> None:
        self.arms = list(arms)
        self.rerank = rerank
        self.rerank_params = rerank_params
        self.type = NULLTYPE


def weighted(distance: ColumnElement, weight: float) -> SearchArm:
    """``weighted(Item.embedding.cosine_distance(qv), 0.7)`` -- one arm
    of a :func:`hybrid_search` call."""
    return SearchArm(distance, weight)


def hybrid_search(
    *arms: SearchArm, rerank: str = "RRF", **rerank_params: t.Any
) -> HybridSearchClause:
    """
    ``select(Item.id).order_by(hybrid_search(
        weighted(Item.embedding.cosine_distance(dense_q), 0.7),
        weighted(Item.sparse.max_inner_product(sparse_q), 0.3),
        rerank="RRF", k=60,
    )).limit(10)``
    """
    return HybridSearchClause(arms, rerank, rerank_params)


__all__ = ["HybridSearchClause", "SearchArm", "hybrid_search", "weighted"]

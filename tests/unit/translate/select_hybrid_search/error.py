"""Error-path coverage for ``HYBRID SEARCH`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_an_arm_scoring_by_bm25_relevance_is_rejected(build_call_helper):
    """The MilvusQL grammar itself accepts ``BM25_SCORE(...)`` as an
    arm (a dense+full-text hybrid is a real query shape), but there is
    no Milvus ``metric_type`` a full-text relevance score maps to --
    this is a real, distinct gap from a malformed arm, not a parser
    rejection."""
    with pytest.raises(milvusql.NotSupportedError, match="arm"):
        build_call_helper(
            "SELECT id FROM items HYBRID SEARCH "
            "(BM25_SCORE(text, :q)) RERANK RRF LIMIT 5",
            {"q": "hello"},
        )


def test_an_unknown_rerank_strategy_is_rejected(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="RERANK"):
        build_call_helper(
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv) "
            "RERANK BOGUS LIMIT 5",
            {"dv": [0.1] * 8},
        )

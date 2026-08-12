"""Error-path coverage for vector-search ``SELECT`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_hybrid_search_is_not_supported_yet(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="HYBRID"):
        build_call_helper(
            "SELECT id FROM items HYBRID SEARCH "
            "(embedding <=> :q WEIGHT 0.7) RERANK RRF LIMIT 5",
            {"q": [0.1] * 8},
        )

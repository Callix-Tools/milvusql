"""Error-path coverage for ``HYBRID SEARCH`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_an_unknown_rerank_strategy_is_rejected(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="RERANK"):
        build_call_helper(
            "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv) "
            "RERANK BOGUS LIMIT 5",
            {"dv": [0.1] * 8},
        )

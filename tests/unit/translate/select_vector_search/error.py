"""Error-path coverage for vector-search ``SELECT`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_order_by_a_non_column_scalar_expression_raises(build_call_helper):
    """A plain scalar ``ORDER BY`` is sorted client-side (see
    ``select_order_by``), but only by a bare column -- an expression
    (``ORDER BY id + 1``) has no natural Milvus field to fetch and sort
    by."""
    with pytest.raises(milvusql.NotSupportedError, match="ORDER BY"):
        build_call_helper("SELECT id FROM items ORDER BY id + 1 LIMIT 5")

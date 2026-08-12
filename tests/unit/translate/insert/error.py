"""Error-path coverage for ``INSERT`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_missing_bind_parameter_raises_programming_error(build_call_helper):
    with pytest.raises(milvusql.ProgrammingError, match="emb"):
        build_call_helper(
            "INSERT INTO items (embedding) VALUES (:emb)",
            {},
        )

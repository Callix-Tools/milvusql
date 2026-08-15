"""Error-path coverage for ``DROP`` dispatch: table, index and
database drops are wired up -- any other ``DROP`` kind is an honest
"not supported", not silently treated as a table drop."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_drop_index_without_a_table_raises_programming_error(
    build_call_helper,
):
    """Milvus scopes index names to a collection, so a bare
    ``DROP INDEX idx`` (which the grammar itself accepts) is
    unanswerable -- the message says what to add."""
    with pytest.raises(milvusql.ProgrammingError, match="ON"):
        build_call_helper("DROP INDEX idx_emb")


def test_an_unwired_drop_kind_raises_not_supported_error(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="DROP"):
        build_call_helper("DROP VIEW v")

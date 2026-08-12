"""Error-path coverage for ``DROP`` dispatch: only ``DROP TABLE`` is
wired up (mirroring ``CREATE TABLE``/``CREATE INDEX``'s own kind check)
-- other ``DROP`` kinds are an honest "not supported", not silently
treated as a table drop."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_drop_index_raises_not_supported_error(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="INDEX"):
        build_call_helper("DROP INDEX idx_emb ON items")


def test_drop_database_raises_not_supported_error(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="DATABASE"):
        build_call_helper("DROP DATABASE mydb")

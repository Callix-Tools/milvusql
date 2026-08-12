"""Integration coverage for connection-lifecycle methods that are
expected to raise (D7): ``rollback()`` is not supported, sync and
async."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


class TestSync:
    def test_rollback_raises_not_supported(self, conn):
        with pytest.raises(milvusql.NotSupportedError):
            conn.rollback()


class TestAsync:
    async def test_rollback_raises_not_supported(self, aconn):
        with pytest.raises(milvusql.NotSupportedError):
            await aconn.rollback()

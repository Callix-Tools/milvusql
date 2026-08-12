"""Error-path integration coverage for ``execute()``, sync and async,
against a real (embedded) Milvus Lite instance."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


class TestSync:
    def test_missing_collection_raises_programming_error(self, conn, cur):
        with pytest.raises(milvusql.ProgrammingError):
            cur.execute("SELECT id FROM does_not_exist LIMIT 1")

    def test_closed_cursor_rejects_execute(self, loaded_items, cur):
        cur.close()
        with pytest.raises(milvusql.InterfaceError):
            cur.execute("SELECT id FROM items LIMIT 1")


class TestAsync:
    async def test_missing_collection_raises_programming_error(self, acur):
        with pytest.raises(milvusql.ProgrammingError):
            await acur.execute("SELECT id FROM does_not_exist LIMIT 1")

    async def test_closed_cursor_rejects_execute(self, acur):
        await acur.close()
        with pytest.raises(milvusql.InterfaceError):
            await acur.execute("SELECT id FROM items LIMIT 1")

"""Integration coverage for connection-lifecycle methods that are
expected to succeed (D7): ``commit()`` is a no-op, sync and async."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


class TestSync:
    def test_commit_is_a_noop(self, conn):
        conn.commit()  # must not raise (D7)

    def test_drop_table_if_exists_on_a_missing_collection_does_not_raise(
        self, cur
    ):
        # See `tests/unit/translate/drop_table/act.py` for why this
        # needs no special-casing in the translator: Milvus's own
        # `drop_collection` is already a no-op on a name that was
        # never there, `IF EXISTS` or not.
        cur.execute("DROP TABLE IF EXISTS does_not_exist_xyz")


class TestAsync:
    async def test_commit_is_a_noop(self, aconn):
        await aconn.commit()  # must not raise (D7)

    async def test_drop_table_if_exists_on_a_missing_collection_does_not_raise(
        self, acur
    ):
        await acur.execute("DROP TABLE IF EXISTS does_not_exist_xyz")

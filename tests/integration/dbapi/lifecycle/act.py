"""Integration coverage for connection-lifecycle methods that are
expected to succeed (D7): ``commit()`` is a no-op, sync and async."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


class TestSync:
    def test_commit_is_a_noop(self, conn):
        conn.commit()  # must not raise (D7)


class TestAsync:
    async def test_commit_is_a_noop(self, aconn):
        await aconn.commit()  # must not raise (D7)

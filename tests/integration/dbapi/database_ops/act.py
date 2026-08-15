"""Integration coverage for database-level DDL (``CREATE``/``USE``/
``DROP DATABASE``) and ``DROP INDEX`` against a real Milvus server."""

from __future__ import annotations

import pytest

from tests.fixtures.dbapi import CREATE_ITEMS, CREATE_ITEMS_INDEX

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


def test_database_lifecycle(conn, milvus_db_name):
    """Create an auxiliary database, switch into it, confirm it is
    empty and listed, switch back, drop it -- the full ``SHOW
    DATABASES``/``CREATE``/``USE``/``DROP DATABASE`` loop."""
    auxiliary = f"{milvus_db_name}_aux"
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE {auxiliary}")
    try:
        cur.execute("SHOW DATABASES")
        assert (auxiliary,) in cur.fetchall()
        cur.execute(f"USE {auxiliary}")
        cur.execute("SHOW TABLES")
        assert cur.fetchall() == []
    finally:
        # Back to the worker's own database before dropping -- and
        # before the per-test cleanup runs against it.
        cur.execute(f"USE {milvus_db_name}")
        cur.execute(f"DROP DATABASE {auxiliary}")
    cur.execute("SHOW DATABASES")
    assert (auxiliary,) not in cur.fetchall()


def test_drop_index_frees_the_name_for_recreation(conn):
    """``DROP INDEX idx ON table`` genuinely removes the index: Milvus
    rejects a second ``CREATE INDEX`` on an already-indexed field, so
    recreating under the same name only succeeds if the drop was
    real."""
    cur = conn.cursor()
    cur.execute(CREATE_ITEMS)
    cur.execute(CREATE_ITEMS_INDEX)
    # An index in use by a loaded collection can't be dropped.
    cur.execute("RELEASE TABLE items")
    cur.execute("DROP INDEX idx_emb ON items")
    cur.execute(CREATE_ITEMS_INDEX)
    cur.execute("LOAD TABLE items")
    cur.execute("SELECT COUNT(*) FROM items")
    assert cur.fetchall() == [(0,)]

"""Integration coverage for ``SHOW TABLES`` / ``SHOW DATABASES`` /
``DESCRIBE`` against a real Milvus server."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


def test_show_tables_sees_a_created_table(loaded_items):
    cur = loaded_items.cursor()
    cur.execute("SHOW TABLES")
    assert ("items",) in cur.fetchall()
    assert cur.description[0][0] == "table_name"


def test_show_databases_lists_at_least_default(conn):
    cur = conn.cursor()
    cur.execute("SHOW DATABASES")
    names = {row[0] for row in cur.fetchall()}
    assert "default" in names


def test_describe_round_trips_the_created_schema(loaded_items):
    cur = loaded_items.cursor()
    cur.execute("DESCRIBE items")
    rows = {row[0]: row for row in cur.fetchall()}
    assert rows["id"] == ("id", "BIGINT", "NO", "PRI", "auto_increment")
    assert rows["embedding"][1] == "VECTOR(8)"
    assert rows["category"][1] == "VARCHAR(64)"


async def test_async_cursor_answers_show_tables_too(aconn):
    cur = aconn.cursor()
    await cur.execute("SHOW TABLES")
    assert ("items",) in await cur.fetchall()

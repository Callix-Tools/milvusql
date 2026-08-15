"""Integration coverage for reading past Milvus's per-call row ceiling
(16384) against a real server -- the primary-key-cursor page loop that
replaced the old ``NotSupportedError``. One collection a few hundred
rows past the ceiling is shared by every test here (building it is the
expensive part), covering the plain unbounded ``SELECT``, the
client-side aggregate and ``COUNT(*)``."""

from __future__ import annotations

import pytest

from milvusql.translate._common import DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

TOTAL_ROWS = DEFAULT_QUERY_LIMIT + 300


@pytest.fixture
def big_collection(conn):
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE big (id BIGINT PRIMARY KEY, n INT, v VECTOR(2)) "
        "WITH (consistency_level='Strong')"
    )
    cur.execute(
        "CREATE INDEX idx_big ON big (v) USING FLAT WITH (metric_type='L2')"
    )
    batch = 4000
    for start in range(0, TOTAL_ROWS, batch):
        cur.executemany(
            "INSERT INTO big (id, n, v) VALUES (:id, :n, :v)",
            [
                {"id": index, "n": index % 7, "v": [0.0, 0.0]}
                for index in range(start, min(start + batch, TOTAL_ROWS))
            ],
        )
    return cur


def test_a_select_with_no_limit_returns_every_row(big_collection):
    big_collection.execute("SELECT id FROM big")
    rows = big_collection.fetchall()
    assert len(rows) == TOTAL_ROWS
    assert len({row[0] for row in rows}) == TOTAL_ROWS


def test_a_client_side_aggregate_covers_every_row(big_collection):
    big_collection.execute("SELECT SUM(n) AS total FROM big")
    (row,) = big_collection.fetchall()
    assert row[0] == sum(index % 7 for index in range(TOTAL_ROWS))


def test_count_star_still_answers_server_side(big_collection):
    big_collection.execute("SELECT COUNT(*) FROM big")
    assert big_collection.fetchall() == [(TOTAL_ROWS,)]


async def test_the_async_cursor_pages_identically(aconn, big_collection):
    cur = aconn.cursor()
    await cur.execute("SELECT id FROM big WHERE id >= :floor", {"floor": 0})
    rows = await cur.fetchall()
    assert len(rows) == TOTAL_ROWS

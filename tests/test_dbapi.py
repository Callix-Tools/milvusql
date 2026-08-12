"""Integration coverage against a real (embedded) Milvus Lite
instance: create/load/insert/search/delete/release through both the
sync ``Cursor`` and the async ``AsyncCursor``, verifying they return
the same shape of answer for the same MilvusQL text."""

from __future__ import annotations

import pytest

import milvusql

EMB_BOOK = [0.1] * 8
EMB_MOVIE = [0.9] * 8

VECTOR_SEARCH_SQL = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <-> :q LIMIT 5 SEARCH PARAMS (ef_search=64)"
)


def _seed(cur):
    cur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_BOOK, "cat": "book"},
    )
    cur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_MOVIE, "cat": "movie"},
    )


async def _aseed(acur):
    await acur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_BOOK, "cat": "book"},
    )
    await acur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_MOVIE, "cat": "movie"},
    )


class TestSync:
    def test_insert_reports_rowcount(self, loaded_items, cur):
        cur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        assert cur.rowcount == 1
        assert cur.description is None

    def test_vector_search_returns_nearest_row(self, loaded_items, cur):
        _seed(cur)
        cur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        assert cur.description == [
            ("id", None, None, None, None, None, True),
            ("category", None, None, None, None, None, True),
        ]
        rows = cur.fetchall()
        assert rows == [(1, "book")]

    def test_plain_filter_select_uses_query_not_search(
        self, loaded_items, cur
    ):
        _seed(cur)
        cur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(cur.fetchall()) == [(1, "book"), (2, "movie")]

    def test_delete_reports_rowcount_and_removes_row(self, loaded_items, cur):
        _seed(cur)
        cur.execute(
            "DELETE FROM items WHERE category = :cat", {"cat": "movie"}
        )
        assert cur.rowcount == 1
        cur.execute("SELECT id FROM items LIMIT 10")
        assert cur.fetchall() == [(1,)]

    def test_fetchone_and_fetchmany_paginate_the_same_rows(
        self, loaded_items, cur
    ):
        _seed(cur)
        cur.execute("SELECT id FROM items LIMIT 10")
        first = cur.fetchone()
        rest = cur.fetchmany(10)
        assert first is not None
        assert first not in rest
        assert cur.fetchone() is None

    def test_commit_is_a_noop(self, conn):
        conn.commit()  # must not raise (D7)

    def test_rollback_raises_not_supported(self, conn):
        with pytest.raises(milvusql.NotSupportedError):
            conn.rollback()

    def test_missing_collection_raises_programming_error(self, conn, cur):
        with pytest.raises(milvusql.ProgrammingError):
            cur.execute("SELECT id FROM does_not_exist LIMIT 1")

    def test_executemany_sums_rowcount(self, loaded_items, cur):
        cur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [
                {"emb": EMB_BOOK, "cat": "book"},
                {"emb": EMB_MOVIE, "cat": "movie"},
            ],
        )
        assert cur.rowcount == 2

    def test_closed_cursor_rejects_execute(self, loaded_items, cur):
        cur.close()
        with pytest.raises(milvusql.InterfaceError):
            cur.execute("SELECT id FROM items LIMIT 1")


class TestAsync:
    async def test_insert_reports_rowcount(self, acur):
        await acur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        assert acur.rowcount == 1
        assert acur.description is None

    async def test_vector_search_returns_nearest_row(self, acur):
        await _aseed(acur)
        await acur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        rows = await acur.fetchall()
        assert rows == [(1, "book")]

    async def test_plain_filter_select_uses_query_not_search(self, acur):
        await _aseed(acur)
        await acur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(await acur.fetchall()) == [(1, "book"), (2, "movie")]

    async def test_delete_reports_rowcount_and_removes_row(self, acur):
        await _aseed(acur)
        await acur.execute(
            "DELETE FROM items WHERE category = :cat", {"cat": "movie"}
        )
        assert acur.rowcount == 1
        await acur.execute("SELECT id FROM items LIMIT 10")
        assert await acur.fetchall() == [(1,)]

    async def test_commit_is_a_noop(self, aconn):
        aconn.commit()  # must not raise (D7)

    async def test_rollback_raises_not_supported(self, aconn):
        with pytest.raises(milvusql.NotSupportedError):
            aconn.rollback()

    async def test_missing_collection_raises_programming_error(self, acur):
        with pytest.raises(milvusql.ProgrammingError):
            await acur.execute("SELECT id FROM does_not_exist LIMIT 1")

    async def test_executemany_sums_rowcount(self, acur):
        await acur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [
                {"emb": EMB_BOOK, "cat": "book"},
                {"emb": EMB_MOVIE, "cat": "movie"},
            ],
        )
        assert acur.rowcount == 2

    async def test_closed_cursor_rejects_execute(self, acur):
        acur.close()
        with pytest.raises(milvusql.InterfaceError):
            await acur.execute("SELECT id FROM items LIMIT 1")


class TestConsistencyLevelFallback:
    """D6: a query's own ``CONSISTENCY LEVEL`` always wins; the
    connection-level default only fills in when a statement doesn't
    set one. Exercised against a stub client so the assertion is on
    the exact kwargs passed, not on Milvus's observable read
    consistency (not something a single-node Lite instance can
    distinguish)."""

    def test_connection_default_applied_when_statement_omits_it(
        self, monkeypatch, loaded_items, conn
    ):
        conn.consistency_level = "Strong"
        cur = conn.cursor()
        seen_kwargs = {}
        real_search = conn._client.search

        def spy_search(**kwargs):
            seen_kwargs.update(kwargs)
            return real_search(**kwargs)

        monkeypatch.setattr(conn._client, "search", spy_search)
        cur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        assert seen_kwargs["consistency_level"] == "Strong"

    def test_statement_level_clause_overrides_connection_default(
        self, monkeypatch, loaded_items, conn
    ):
        conn.consistency_level = "Strong"
        cur = conn.cursor()
        seen_kwargs = {}
        real_search = conn._client.search

        def spy_search(**kwargs):
            seen_kwargs.update(kwargs)
            return real_search(**kwargs)

        monkeypatch.setattr(conn._client, "search", spy_search)
        cur.execute(
            VECTOR_SEARCH_SQL + " CONSISTENCY LEVEL Eventually",
            {"cat": "book", "q": EMB_BOOK},
        )
        assert seen_kwargs["consistency_level"] == "Eventually"

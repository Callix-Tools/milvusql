"""Integration coverage against a real (embedded) Milvus Lite
instance: insert/search/select/delete through both the sync
``Cursor`` and the async ``AsyncCursor``, verifying they return the
same shape of answer for the same MilvusQL text."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

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

    def test_executemany_sums_rowcount(self, loaded_items, cur):
        cur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [
                {"emb": EMB_BOOK, "cat": "book"},
                {"emb": EMB_MOVIE, "cat": "movie"},
            ],
        )
        assert cur.rowcount == 2


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

    async def test_executemany_sums_rowcount(self, acur):
        await acur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [
                {"emb": EMB_BOOK, "cat": "book"},
                {"emb": EMB_MOVIE, "cat": "movie"},
            ],
        )
        assert acur.rowcount == 2


class TestUpdate:
    """``UPDATE`` against a real Milvus Lite instance: a genuine
    read-then-``upsert`` round trip (see ``Call.then``'s docstring),
    not a single RPC -- worth exercising end to end, not just at the
    ``build_call`` unit level."""

    def test_update_changes_only_matching_rows(self, loaded_items, cur):
        _seed(cur)
        cur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "book"},
        )
        assert cur.rowcount == 1
        cur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(cur.fetchall()) == [(1, "novel"), (2, "movie")]

    def test_update_preserves_the_vector_field_it_did_not_set(
        self, loaded_items, cur
    ):
        _seed(cur)
        cur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "book"},
        )
        cur.execute(
            "SELECT id, category FROM items WHERE category = :cat "
            "ORDER BY embedding <-> :q LIMIT 5",
            {"cat": "novel", "q": EMB_BOOK},
        )
        assert cur.fetchall() == [(1, "novel")]

    def test_update_matching_nothing_reports_zero_rowcount(
        self, loaded_items, cur
    ):
        _seed(cur)
        cur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "nonexistent"},
        )
        assert cur.rowcount == 0

    async def test_async_update_changes_only_matching_rows(self, acur):
        await _aseed(acur)
        await acur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "book"},
        )
        assert acur.rowcount == 1
        await acur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(await acur.fetchall()) == [(1, "novel"), (2, "movie")]


class TestAggregate:
    def test_count_star_counts_every_matching_row(self, loaded_items, cur):
        _seed(cur)
        cur.execute(
            'SELECT COUNT(*) AS "n" FROM items WHERE category = :c',
            {"c": "book"},
        )
        assert cur.fetchall() == [(1,)]

    def test_sum_reduces_a_real_column(self, loaded_items, cur):
        _seed(cur)
        cur.execute('SELECT SUM("id") AS "total" FROM items')
        assert cur.fetchall() == [(3,)]


class TestScalarOrderByAndInFilter:
    def test_order_by_a_plain_column_sorts_client_side(
        self, loaded_items, cur
    ):
        _seed(cur)
        cur.execute("SELECT id, category FROM items ORDER BY id DESC LIMIT 10")
        assert cur.fetchall() == [(2, "movie"), (1, "book")]

    def test_in_filter_matches_only_the_listed_values(self, loaded_items, cur):
        _seed(cur)
        cur.execute(
            'SELECT id FROM items WHERE "category" IN (:a, :b) LIMIT 10',
            {"a": "movie", "b": "nonexistent"},
        )
        assert cur.fetchall() == [(2,)]


class TestNoExplicitLimit:
    def test_a_select_with_no_limit_clause_returns_more_than_ten_rows(
        self, loaded_items, cur
    ):
        """Regression test: a bare ``SELECT`` with no ``LIMIT`` used to
        silently cap at 10 rows with no error at all."""
        cur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [{"emb": EMB_BOOK, "cat": f"cat{i}"} for i in range(25)],
        )
        cur.execute("SELECT id FROM items")
        assert len(cur.fetchall()) == 25

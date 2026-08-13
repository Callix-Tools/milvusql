"""Integration coverage against a real Milvus server (via
``testcontainers``, see ``tests/fixtures/containers/milvus_server.py``):
insert/search/select/delete through both the sync ``Cursor`` and the
async ``AsyncCursor``, verifying they return the same shape of answer
for the same MilvusQL text."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

EMB_BOOK = [0.1] * 8
EMB_MOVIE = [0.9] * 8

#: `<=>` (cosine distance), not `<->` (L2): `CREATE_ITEMS_INDEX`
#: (``tests/fixtures/dbapi.py``) builds its index with `metric_type=
#: 'COSINE'` -- a real (non-Lite) server rejects a search whose own
#: metric doesn't match the index's (confirmed directly: "metric type
#: not match: invalid parameter[expected=COSINE][actual=L2]"), so this
#: has to ask for the same metric the index actually has.
VECTOR_SEARCH_SQL = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <=> :q LIMIT 5 SEARCH PARAMS (ef_search=64)"
)


def _seed(cur):
    cur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_BOOK, "cat": "book"},
    )
    book_id = cur.lastrowid
    cur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_MOVIE, "cat": "movie"},
    )
    movie_id = cur.lastrowid
    return book_id, movie_id


async def _aseed(acur):
    await acur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_BOOK, "cat": "book"},
    )
    book_id = acur.lastrowid
    await acur.execute(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": EMB_MOVIE, "cat": "movie"},
    )
    movie_id = acur.lastrowid
    return book_id, movie_id


class TestSync:
    def test_insert_reports_rowcount(self, loaded_items, cur):
        cur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        assert cur.rowcount == 1
        assert cur.description is None

    def test_vector_search_returns_nearest_row(self, loaded_items, cur):
        book_id, _movie_id = _seed(cur)
        cur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        assert cur.description == [
            ("id", None, None, None, None, None, True),
            ("category", None, None, None, None, None, True),
        ]
        rows = cur.fetchall()
        assert rows == [(book_id, "book")]

    def test_plain_filter_select_uses_query_not_search(
        self, loaded_items, cur
    ):
        book_id, movie_id = _seed(cur)
        cur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(cur.fetchall()) == sorted(
            [(book_id, "book"), (movie_id, "movie")]
        )

    def test_delete_reports_rowcount_and_removes_row(self, loaded_items, cur):
        book_id, _movie_id = _seed(cur)
        cur.execute(
            "DELETE FROM items WHERE category = :cat", {"cat": "movie"}
        )
        assert cur.rowcount == 1
        cur.execute("SELECT id FROM items LIMIT 10")
        assert cur.fetchall() == [(book_id,)]

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


class TestExecutemanyBatching:
    """``executemany()`` over ``INSERT`` merges every parameter set into
    one ``insert()`` RPC (see ``build_batch_call``) instead of one RPC
    per row -- verified end to end here, not just at the
    ``build_batch_call`` unit level, since the interesting part is that
    the sync ``Cursor`` actually takes the batched path."""

    def test_insert_executemany_issues_a_single_rpc(
        self, loaded_items, cur, monkeypatch
    ):
        calls = []
        real_insert = cur.connection._client.insert

        def counting_insert(*args, **kwargs):
            calls.append((args, kwargs))
            return real_insert(*args, **kwargs)

        monkeypatch.setattr(cur.connection._client, "insert", counting_insert)
        cur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [{"emb": EMB_BOOK, "cat": f"cat{i}"} for i in range(5)],
        )
        assert cur.rowcount == 5
        assert len(calls) == 1
        assert len(calls[0][1]["data"]) == 5

    def test_lastrowid_is_the_last_row_across_every_parameter_set(
        self, loaded_items, cur
    ):
        cur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [{"emb": EMB_BOOK, "cat": f"cat{i}"} for i in range(3)],
        )
        # Capture before the follow-up SELECT below, which -- correctly
        # -- resets `lastrowid` back to `None` (see `_query_rows`):
        # `lastrowid` only ever means "what the most recent INSERT
        # assigned," not "whatever it once was."
        lastrowid = cur.lastrowid
        rows = cur.execute(
            "SELECT id FROM items ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert lastrowid == rows[0][0]

    def test_an_empty_parameter_sequence_is_a_no_op(self, loaded_items, cur):
        cur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)", []
        )
        assert cur.rowcount == 0

    def test_update_executemany_still_falls_back_to_one_rpc_per_row(
        self, loaded_items, cur
    ):
        """``UPDATE`` has no batched primitive (``build_batch_call``
        returns ``None`` for it) -- ``executemany`` must still work,
        just via the pre-existing one-``execute()``-per-parameter-set
        loop."""
        book_id, movie_id = _seed(cur)
        cur.executemany(
            "UPDATE items SET category = :new WHERE category = :old",
            [
                {"new": "novel", "old": "book"},
                {"new": "film", "old": "movie"},
            ],
        )
        assert cur.rowcount == 2
        cur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(cur.fetchall()) == sorted(
            [(book_id, "novel"), (movie_id, "film")]
        )


class TestAsync:
    async def test_insert_reports_rowcount(self, acur):
        await acur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        assert acur.rowcount == 1
        assert acur.description is None

    async def test_vector_search_returns_nearest_row(self, acur):
        book_id, _movie_id = await _aseed(acur)
        await acur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        rows = await acur.fetchall()
        assert rows == [(book_id, "book")]

    async def test_plain_filter_select_uses_query_not_search(self, acur):
        book_id, movie_id = await _aseed(acur)
        await acur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(await acur.fetchall()) == sorted(
            [(book_id, "book"), (movie_id, "movie")]
        )

    async def test_delete_reports_rowcount_and_removes_row(self, acur):
        book_id, _movie_id = await _aseed(acur)
        await acur.execute(
            "DELETE FROM items WHERE category = :cat", {"cat": "movie"}
        )
        assert acur.rowcount == 1
        await acur.execute("SELECT id FROM items LIMIT 10")
        assert await acur.fetchall() == [(book_id,)]

    async def test_executemany_sums_rowcount(self, acur):
        await acur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [
                {"emb": EMB_BOOK, "cat": "book"},
                {"emb": EMB_MOVIE, "cat": "movie"},
            ],
        )
        assert acur.rowcount == 2

    async def test_insert_executemany_issues_a_single_rpc(
        self, acur, monkeypatch
    ):
        calls = []
        real_insert = acur.connection._client.insert

        async def counting_insert(*args, **kwargs):
            calls.append((args, kwargs))
            return await real_insert(*args, **kwargs)

        monkeypatch.setattr(acur.connection._client, "insert", counting_insert)
        await acur.executemany(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            [{"emb": EMB_BOOK, "cat": f"cat{i}"} for i in range(5)],
        )
        assert acur.rowcount == 5
        assert len(calls) == 1
        assert len(calls[0][1]["data"]) == 5


class TestUpdate:
    """``UPDATE`` against a real Milvus Lite instance: a genuine
    read-then-``upsert`` round trip (see ``Call.then``'s docstring),
    not a single RPC -- worth exercising end to end, not just at the
    ``build_call`` unit level."""

    def test_update_changes_only_matching_rows(self, loaded_items, cur):
        book_id, movie_id = _seed(cur)
        cur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "book"},
        )
        assert cur.rowcount == 1
        cur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(cur.fetchall()) == sorted(
            [(book_id, "novel"), (movie_id, "movie")]
        )

    def test_update_preserves_the_vector_field_it_did_not_set(
        self, loaded_items, cur
    ):
        book_id, _movie_id = _seed(cur)
        cur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "book"},
        )
        cur.execute(
            "SELECT id, category FROM items WHERE category = :cat "
            "ORDER BY embedding <=> :q LIMIT 5",
            {"cat": "novel", "q": EMB_BOOK},
        )
        assert cur.fetchall() == [(book_id, "novel")]

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
        book_id, movie_id = await _aseed(acur)
        await acur.execute(
            "UPDATE items SET category = :new WHERE category = :old",
            {"new": "novel", "old": "book"},
        )
        assert acur.rowcount == 1
        await acur.execute("SELECT id, category FROM items LIMIT 10")
        assert sorted(await acur.fetchall()) == sorted(
            [(book_id, "novel"), (movie_id, "movie")]
        )


class TestAggregate:
    def test_count_star_counts_every_matching_row(self, loaded_items, cur):
        _seed(cur)
        cur.execute(
            'SELECT COUNT(*) AS "n" FROM items WHERE category = :c',
            {"c": "book"},
        )
        assert cur.fetchall() == [(1,)]

    def test_sum_reduces_a_real_column(self, loaded_items, cur):
        book_id, movie_id = _seed(cur)
        cur.execute('SELECT SUM("id") AS "total" FROM items')
        assert cur.fetchall() == [(book_id + movie_id,)]


class TestScalarOrderByAndInFilter:
    def test_order_by_a_plain_column_sorts_client_side(
        self, loaded_items, cur
    ):
        book_id, movie_id = _seed(cur)
        cur.execute("SELECT id, category FROM items ORDER BY id DESC LIMIT 10")
        assert cur.fetchall() == [(movie_id, "movie"), (book_id, "book")]

    def test_in_filter_matches_only_the_listed_values(self, loaded_items, cur):
        _book_id, movie_id = _seed(cur)
        cur.execute(
            'SELECT id FROM items WHERE "category" IN (:a, :b) LIMIT 10',
            {"a": "movie", "b": "nonexistent"},
        )
        assert cur.fetchall() == [(movie_id,)]


class TestLikeIsNullBetween:
    """``LIKE``/``IS [NOT] NULL``/``[NOT] BETWEEN`` against a real
    Milvus Lite instance -- worth exercising end to end, not just at
    the ``build_call`` unit level, since ``BETWEEN`` in particular
    transpiles to a shape Milvus's filter DSL never sees literally."""

    def test_like_matches_a_prefix_pattern(self, loaded_items, cur):
        book_id, _movie_id = _seed(cur)
        cur.execute(
            "SELECT id FROM items WHERE category LIKE :pat LIMIT 10",
            {"pat": "boo%"},
        )
        assert cur.fetchall() == [(book_id,)]

    def test_not_like_excludes_a_prefix_pattern(self, loaded_items, cur):
        _book_id, movie_id = _seed(cur)
        cur.execute(
            "SELECT id FROM items WHERE category NOT LIKE :pat LIMIT 10",
            {"pat": "boo%"},
        )
        assert cur.fetchall() == [(movie_id,)]

    def test_is_null_matches_only_the_null_row(self, loaded_items, cur):
        _seed(cur)
        cur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": None},
        )
        null_id = cur.lastrowid
        cur.execute("SELECT id FROM items WHERE category IS NULL LIMIT 10")
        assert cur.fetchall() == [(null_id,)]

    def test_is_not_null_excludes_the_null_row(self, loaded_items, cur):
        book_id, movie_id = _seed(cur)
        cur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": None},
        )
        cur.execute("SELECT id FROM items WHERE category IS NOT NULL LIMIT 10")
        assert sorted(cur.fetchall()) == sorted([(book_id,), (movie_id,)])

    def test_between_matches_an_inclusive_range(self, loaded_items, cur):
        book_id, _movie_id = _seed(cur)
        cur.execute(
            "SELECT id FROM items WHERE id BETWEEN :lo AND :hi LIMIT 10",
            {"lo": book_id, "hi": book_id},
        )
        assert cur.fetchall() == [(book_id,)]

    def test_not_between_excludes_the_range(self, loaded_items, cur):
        book_id, movie_id = _seed(cur)
        cur.execute(
            "SELECT id FROM items WHERE id NOT BETWEEN :lo AND :hi LIMIT 10",
            {"lo": book_id, "hi": book_id},
        )
        assert cur.fetchall() == [(movie_id,)]


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

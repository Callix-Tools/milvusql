"""D6: a query's own ``CONSISTENCY LEVEL`` always wins; the
connection-level default only fills in when a statement doesn't set
one. Exercised against a real Milvus server (via ``testcontainers``),
but the assertion is still on the exact ``consistency_level`` kwarg
``search()`` was actually called with (via a spy wrapping the real
call), not on Milvus's observable read consistency -- the same
staleness window this exercises is exactly what genuine eventual
consistency makes unobservable/nondeterministic to assert on
directly."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

EMB_BOOK = [0.1] * 8

#: `<=>` (cosine distance), not `<->` (L2): `CREATE_ITEMS_INDEX`
#: (``tests/fixtures/dbapi.py``) builds its index with `metric_type=
#: 'COSINE'` -- a real (non-Lite) server rejects a search whose own
#: metric doesn't match the index's.
VECTOR_SEARCH_SQL = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <=> :q LIMIT 5 SEARCH PARAMS (ef_search=64)"
)


class TestConsistencyLevelFallback:
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

"""Integration coverage for auto-``LOAD`` (D2 revised, see
``milvusql._shared.note_load_state`` and
``Connection._loaded_collections``): a ``search``/``query``/
``hybrid_search`` transparently loads its target collection the first
time a connection touches it, instead of requiring a caller-issued
``LOAD TABLE`` first. Exercised against a real Milvus server (via
``testcontainers``) -- the "not loaded" failure this replaces is a
real server-reported error, not something a mock can stand in for."""

from __future__ import annotations

import pytest

from milvusql import aio
from tests.fixtures.dbapi import CREATE_ITEMS, CREATE_ITEMS_INDEX

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

EMB_BOOK = [0.1] * 8

#: `<=>` (cosine distance) to match `CREATE_ITEMS_INDEX`'s
#: `metric_type='COSINE'` (`tests/fixtures/dbapi.py`) -- see
#: `tests/integration/dbapi/execute/act.py`'s own `VECTOR_SEARCH_SQL`
#: for why L2 (`<->`) would be rejected by a real server.
VECTOR_SEARCH_SQL = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <=> :q LIMIT 5 SEARCH PARAMS (ef_search=64)"
)


@pytest.fixture
def indexed_items(conn, cur):
    """Same shape as `tests/fixtures/dbapi.py`'s `loaded_items`, minus
    the explicit `LOAD TABLE` -- the whole point of this module is
    that queries against this fixture still work without it."""
    cur.execute(CREATE_ITEMS)
    cur.execute(CREATE_ITEMS_INDEX)
    return conn


class TestSync:
    def test_select_succeeds_without_a_prior_load_table(
        self, indexed_items, cur
    ):
        cur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        cur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        rows = cur.fetchall()
        assert len(rows) == 1

    def test_second_query_does_not_reissue_load_collection(
        self, monkeypatch, indexed_items, cur
    ):
        calls = []
        real_load = indexed_items._client.load_collection

        def spy_load(*args, **kwargs):
            calls.append((args, kwargs))
            return real_load(*args, **kwargs)

        monkeypatch.setattr(indexed_items._client, "load_collection", spy_load)
        cur.execute("SELECT id FROM items LIMIT 1")
        cur.execute("SELECT id FROM items LIMIT 1")
        cur.execute("SELECT id FROM items LIMIT 1")
        assert len(calls) == 1

    def test_insert_alone_never_triggers_a_load(
        self, monkeypatch, indexed_items, cur
    ):
        calls = []
        real_load = indexed_items._client.load_collection
        monkeypatch.setattr(
            indexed_items._client,
            "load_collection",
            lambda *a, **k: calls.append((a, k)) or real_load(*a, **k),
        )
        cur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        assert calls == []

    def test_release_table_makes_the_next_query_reload(
        self, indexed_items, cur
    ):
        cur.execute("SELECT id FROM items LIMIT 1")
        assert "items" in indexed_items._loaded_collections
        cur.execute("RELEASE TABLE items")
        assert "items" not in indexed_items._loaded_collections
        # The next query must still succeed -- it has to reload behind
        # the scenes rather than surface Milvus's "not loaded" error.
        cur.execute("SELECT id FROM items LIMIT 1")
        assert "items" in indexed_items._loaded_collections

    def test_explicit_load_table_is_still_honored_and_updates_the_cache(
        self, indexed_items, cur
    ):
        cur.execute("LOAD TABLE items")
        assert "items" in indexed_items._loaded_collections
        cur.execute("SELECT id FROM items LIMIT 1")


@pytest.fixture
async def unloaded_aconn(db_uri, milvus_db_name, indexed_items):
    """The async mirror of `indexed_items`: a fresh `AsyncConnection`
    whose own `_loaded_collections` cache starts empty. Depends on
    `indexed_items`, not `loaded_items`, specifically to avoid the
    explicit `LOAD TABLE` `loaded_items` issues -- that runs through
    the *sync* `conn`, so it would prove nothing about whether this
    connection's own auto-LOAD path actually fires."""
    connection = aio.connect(
        uri=db_uri,
        token="root:Milvus",  # noqa: S106 -- well-known Milvus default, not a secret
        db_name=milvus_db_name,
    )
    yield connection
    await connection.close()


class TestAsync:
    async def test_select_succeeds_without_a_prior_load_table(
        self, unloaded_aconn
    ):
        acur = unloaded_aconn.cursor()
        await acur.execute(
            "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
            {"emb": EMB_BOOK, "cat": "book"},
        )
        await acur.execute(VECTOR_SEARCH_SQL, {"cat": "book", "q": EMB_BOOK})
        rows = await acur.fetchall()
        assert len(rows) == 1

    async def test_second_query_does_not_reissue_load_collection(
        self, monkeypatch, unloaded_aconn
    ):
        acur = unloaded_aconn.cursor()
        calls = []
        real_load = unloaded_aconn._client.load_collection

        async def spy_load(*args, **kwargs):
            calls.append((args, kwargs))
            return await real_load(*args, **kwargs)

        monkeypatch.setattr(
            unloaded_aconn._client, "load_collection", spy_load
        )
        await acur.execute("SELECT id FROM items LIMIT 1")
        await acur.execute("SELECT id FROM items LIMIT 1")
        assert len(calls) == 1

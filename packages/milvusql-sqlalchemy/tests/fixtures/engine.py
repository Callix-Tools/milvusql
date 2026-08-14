"""Fixtures for the milvusql-sqlalchemy test suite: every integration
test in this package runs against one REAL Milvus server (a
``testcontainers``-started container, or ``MILVUS_TEST_URI`` --
see ``tests/fixtures/containers/milvus_remote.py``) shared across the
whole pytest-xdist worker process. Isolation across workers is a
dedicated Milvus *database* per worker id; isolation across tests
*within* one worker is a per-test collection-drop teardown
(``_milvus_worker_cleanup``, pulled in transitively through
``db_uri``) -- this replaces what a fresh Milvus Lite on-disk file per
test used to guarantee, so ``pytest-randomly`` can still freely
reorder tests without collection-name collisions."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.fixture
def db_uri(milvus_sync_url: str, _milvus_worker_cleanup) -> str:
    """The real ``milvusql://root:Milvus@host:port/db`` URL string,
    with this test's post-run collection cleanup already wired in as a
    dependency -- so any engine built from this string directly (not
    just through ``engine``/``seeded_engine`` below), e.g.
    ``consistency_level_regression/act.py``'s own
    ``create_engine(db_uri, isolation_level=...)``, still gets a clean
    slate for the next test."""
    return milvus_sync_url


@pytest.fixture
def engine(db_uri):
    # 'Strong', not Milvus's own 'Bounded' default: against a real
    # (non-Lite) server, 'Bounded' permits exactly the staleness
    # window its name promises, so a query issued immediately after an
    # insert/commit in the same test -- including SQLAlchemy's own
    # ORM `Session` refreshing an object's attributes right after
    # `commit()` -- can legitimately not see it yet. Confirmed
    # directly: `orm_autoincrement_insert`'s `Session.add()` +
    # `commit()` intermittently raised `ObjectDeletedError` against a
    # real server with no isolation_level set (never observable
    # against Milvus Lite, which has no replication lag to be
    # inconsistent about). A test that specifically wants to exercise
    # 'Bounded' itself (``consistency_level_regression``) builds its
    # own engine rather than using this fixture.
    eng = create_engine(db_uri, isolation_level="Strong")
    yield eng
    eng.dispose()


@pytest.fixture
def seeded_engine(engine):
    """A real ``items`` collection: created, HNSW-indexed, loaded, and
    holding one row -- everything a reflection or vector-search
    assertion needs, built once through the dialect end-to-end rather
    than per test."""
    metadata = MetaData()
    items = Table(
        "items",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("category", String(64)),
        Column("embedding", VECTOR(8)),
        milvusql_shards=1,
        # 'Strong', not the more realistic 'Bounded': against a real
        # (non-Lite) server, 'Bounded' permits exactly the staleness
        # window its name promises, so a query issued immediately
        # after an insert in the same test can legitimately not see
        # it yet -- confirmed directly, this collection used to say
        # 'Bounded' and every test relying on read-your-writes
        # intermittently got an empty/stale result against a real
        # server (never observable against Milvus Lite, which has no
        # replication lag to be inconsistent about).
        milvusql_consistency_level="Strong",
    )
    metadata.create_all(engine)
    Index(
        "idx_emb",
        items.c.embedding,
        milvusql_using="HNSW",
        milvusql_with={
            "metric_type": "COSINE",
            "M": 16,
            "ef_construction": 200,
        },
    ).create(engine)
    with engine.connect() as conn:
        conn.execute(
            insert(items), [{"category": "book", "embedding": [0.1] * 8}]
        )
        conn.commit()
        # CREATE INDEX doesn't LOAD -- a search against an unloaded
        # collection fails, so this is done explicitly through the raw
        # client (no SQL-text equivalent routed through execute()).
        conn.connection.dbapi_connection._client.load_collection("items")
    return engine, items


@pytest.fixture
async def seeded_engine_aio(seeded_engine):
    """The exact same collection as ``seeded_engine`` -- created,
    HNSW-indexed, and loaded through the *sync* engine, then reopened
    through ``create_async_engine`` for the async-path test to use.
    ``ast_to_pymilvus``'s ``_build_create_index`` docstring notes the
    ``AllocTimestamp``-completion-wait gap in ``CREATE INDEX`` is
    specifically a Milvus *Lite* async-gRPC-server limitation, not a
    general one -- so against this package's real server it may well
    work fine through the async engine directly, but this fixture
    keeps the already-proven sync-build-then-reopen path rather than
    switching without having independently re-verified the async path
    end to end. Both clients talk to the same real server/database,
    same reasoning as the root package's own ``loaded_items``
    fixture."""
    sync_engine, items = seeded_engine
    aio_engine = create_async_engine(
        sync_engine.url.set(drivername="milvusql+aio")
    )
    yield aio_engine, items
    await aio_engine.dispose()

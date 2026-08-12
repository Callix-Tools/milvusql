"""Fixtures for the milvusql-sqlalchemy test suite: every integration
test gets its own Milvus Lite instance (via ``tmp_path``), so
``pytest-randomly`` can freely reorder tests without collection-name
collisions."""

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
def db_uri(tmp_path) -> str:
    return f"milvusql:///{tmp_path}/milvus.db"


@pytest.fixture
def engine(db_uri):
    eng = create_engine(db_uri)
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
        milvusql_consistency_level="Bounded",
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
    """The exact same on-disk collection as ``seeded_engine`` --
    created, HNSW-indexed, and loaded through the *sync* engine, then
    reopened through ``create_async_engine`` for the async-path test to
    use. Not built through the async engine directly: ``CREATE INDEX``
    cannot be exercised through ``milvusql+aio`` against Milvus Lite --
    the ``AllocTimestamp`` RPC its completion-wait calls into isn't
    implemented on Milvus Lite's *async* gRPC server (confirmed
    directly; the sync server handles the equivalent sync call fine,
    already documented in the root package's
    ``ast_to_pymilvus._build_create_index`` docstring). Milvus Lite is
    one on-disk server -- both clients see the same collection, same
    reasoning as the root package's own ``loaded_items`` fixture."""
    sync_engine, items = seeded_engine
    aio_engine = create_async_engine(
        sync_engine.url.set(drivername="milvusql+aio")
    )
    yield aio_engine, items
    await aio_engine.dispose()

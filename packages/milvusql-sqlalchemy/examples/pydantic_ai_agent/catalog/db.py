"""Schema and engines for the `products` collection.

`bootstrap_schema()` runs through a *sync* engine, even though the
agent's tools (`catalog.agent`) are async -- `CREATE INDEX` waits for
completion via an RPC that Milvus Lite's async gRPC server doesn't
implement (a Lite-only gap; a real server's async client handles it
fine -- see `ast_to_pymilvus._build_create_index`'s own docstring in
the root `milvusql` package). Building the schema through the sync
client and only switching to async for the agent's queries sidesteps
that gap entirely, and is the same pattern this project's own async
test suite uses (`milvusql-sqlalchemy`'s `seeded_engine_aio` fixture).
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from catalog.config import (
    COLLECTION_NAME,
    DATABASE_URL,
    SYNC_DATABASE_URL,
    VECTOR_DIM,
)
from milvusql_sqlalchemy.types import VECTOR

metadata = MetaData()

products = Table(
    COLLECTION_NAME,
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("name", String(128)),
    Column("description", String(512)),
    Column("category", String(64)),
    Column("price", Float),
    Column("embedding", VECTOR(VECTOR_DIM)),
    milvusql_shards=1,
    milvusql_consistency_level="Strong",
)

EMBEDDING_INDEX_NAME = "idx_products_embedding"


def bootstrap_schema() -> None:
    """Create the collection, index it, and load it. Safe to call
    repeatedly: each step is guarded so re-running it against an
    already-bootstrapped database is a no-op, not an error. This is a
    demo convenience, not a migration story -- a real deployment
    should manage schema changes through Alembic instead (see the
    `milvusql-sqlalchemy` docs' Alembic guide)."""
    engine = create_engine(SYNC_DATABASE_URL)
    try:
        with engine.begin() as conn:
            metadata.create_all(conn, checkfirst=True)

        with engine.connect() as conn:
            # Checked by column, not by name: Milvus doesn't reliably
            # preserve the literal name `CREATE INDEX` was given (the
            # same gotcha the root test suite's own
            # `async_engine_roundtrip` notes), so a name comparison
            # here would recreate the index -- and fail with "index
            # already exists" -- on every second call.
            existing_indexes = inspect(conn).get_indexes(COLLECTION_NAME)
            if not any(
                ix["column_names"] == ["embedding"]
                for ix in existing_indexes
            ):
                Index(
                    EMBEDDING_INDEX_NAME,
                    products.c.embedding,
                    milvusql_using="HNSW",
                    milvusql_with={
                        "metric_type": "COSINE",
                        "M": 16,
                        "ef_construction": 200,
                    },
                ).create(conn)
            # No SQL-level equivalent routed through `conn.execute()`
            # in this dialect -- go through the raw pymilvus client
            # underneath, same as the package's own test fixtures do
            # (`tests/fixtures/engine.py`'s `seeded_engine`).
            dbapi_connection = conn.connection.dbapi_connection
            dbapi_connection._client.load_collection(COLLECTION_NAME)  # noqa: SLF001
    finally:
        engine.dispose()


def make_async_engine() -> AsyncEngine:
    return create_async_engine(DATABASE_URL)

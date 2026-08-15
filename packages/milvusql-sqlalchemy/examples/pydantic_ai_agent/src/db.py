"""Schema for the `products` collection.

Runs entirely through `milvusql+aio`, the same async engine the
agent's tools (`agent.py`) use: a real Milvus server (this example's
`docker-compose.yaml` target, not Milvus Lite) has no async gap around
`CREATE INDEX`'s completion RPC -- that gap is Milvus-Lite-specific
(see `ast_to_pymilvus._build_create_index`'s own docstring in the root
`milvusql` package). No explicit `LOAD TABLE`/`load_collection()`
either: `search`/`query` auto-load their target collection the first
time a connection touches it (D2 revised, `milvusql.dbapi.Cursor`/
`aio.AsyncCursor`).
"""

from __future__ import annotations

from config import COLLECTION_NAME, DATABASE_URL, VECTOR_DIM
from sqlalchemy import BigInteger, Column, Float, Index, MetaData, String, Table, inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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


def make_engine() -> AsyncEngine:
    return create_async_engine(DATABASE_URL)


async def bootstrap_schema(engine: AsyncEngine) -> None:
    """Create the collection and index it. Safe to call repeatedly:
    each step is guarded so re-running it against an already-
    bootstrapped database is a no-op, not an error. This is a demo
    convenience, not a migration story -- a real deployment should
    manage schema changes through Alembic instead (see the
    `milvusql-sqlalchemy` docs' Alembic guide)."""
    async with engine.begin() as conn:
        # `checkfirst=True` is `create_all`'s own default -- makes
        # this a no-op against a collection that already exists.
        await conn.run_sync(metadata.create_all)

        # Checked by column, not by name: Milvus doesn't reliably
        # preserve the literal name `CREATE INDEX` was given (the same
        # gotcha the root test suite's own `async_engine_roundtrip`
        # notes), so a name comparison here would recreate the index
        # -- and fail with "index already exists" -- on every second
        # call.
        existing_indexes = await conn.run_sync(
            lambda c: inspect(c).get_indexes(COLLECTION_NAME)
        )
        if not any(
            ix["column_names"] == ["embedding"] for ix in existing_indexes
        ):
            await conn.run_sync(
                lambda c: Index(
                    EMBEDDING_INDEX_NAME,
                    products.c.embedding,
                    milvusql_using="HNSW",
                    milvusql_with={
                        "metric_type": "COSINE",
                        "M": 16,
                        "ef_construction": 200,
                    },
                ).create(c)
            )

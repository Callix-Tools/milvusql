"""Coverage proving the exact URL shape a real deployment needs
actually works end to end: ``milvusql+aio://user:pass@host:port/db``
-- host, port, *and* a ``user:password`` credential pair all carried
through ``MilvusDialect_aio.create_connect_args`` to a real Milvus
server (started via ``testcontainers``, or pointed at
``MILVUS_TEST_URI`` -- see
``tests/fixtures/containers/milvus_remote.py``).

Every other integration test in this package builds its engine
through ``tests/fixtures/engine.py``'s ``db_uri``/``engine`` chain,
which also points at the same real server/database -- this module
still keeps its own throwaway, uuid-suffixed collection (created and
dropped within the test itself) rather than reusing ``seeded_engine``/
``seeded_engine_aio``, so it stays self-contained and unaffected by
whatever else is running concurrently in this worker's database."""

from __future__ import annotations

import uuid

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    MetaData,
    String,
    Table,
    select,
)

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


def _items_table(metadata: MetaData, name: str) -> Table:
    return Table(
        name,
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("category", String(64)),
        Column("embedding", VECTOR(8)),
        milvusql_shards=1,
        milvusql_consistency_level="Strong",
    )


async def test_credentials_host_and_port_reach_a_real_server(
    remote_engine_aio,
):
    engine = remote_engine_aio
    # A unique name per run: a real, persistent server (unlike a
    # per-test Milvus-Lite file) can retain collections across runs.
    name = f"items_{uuid.uuid4().hex}"
    metadata = MetaData()
    items = _items_table(metadata, name)

    async with engine.connect() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.run_sync(
            lambda c: Index(
                f"idx_{name}",
                items.c.embedding,
                milvusql_using="HNSW",
                milvusql_with={
                    "metric_type": "COSINE",
                    "M": 16,
                    "ef_construction": 200,
                },
            ).create(c)
        )
        await conn.commit()

    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                items.insert(), [{"category": "book", "embedding": [0.1] * 8}]
            )
            (generated_id,) = result.inserted_primary_key
            await conn.commit()
            # CREATE INDEX doesn't LOAD -- same gotcha as
            # ``seeded_engine``'s fixture docstring.
            raw = await conn.get_raw_connection()
            await raw.driver_connection._client.load_collection(name)

        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(items.c.id, items.c.category).where(
                        items.c.category == "book"
                    )
                )
            ).all()
        # Milvus's real `auto_id` allocator assigns large,
        # timestamp-derived ids -- not the small sequential ones
        # Milvus Lite happened to hand out -- so this compares against
        # whatever id the insert above actually got back, not a
        # literal.
        assert rows == [(generated_id, "book")]
    finally:
        async with engine.connect() as conn:
            await conn.run_sync(items.drop)
            await conn.commit()

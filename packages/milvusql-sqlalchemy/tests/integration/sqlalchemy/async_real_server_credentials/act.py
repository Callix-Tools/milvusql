"""Validation coverage proving the exact URL shape a real deployment
needs actually works end to end: ``milvusql+aio://user:pass@host:port/db``
-- host, port, *and* a ``user:password`` credential pair all carried
through ``MilvusDialect_aio.create_connect_args`` to a real, non-Lite
Milvus server (started via ``testcontainers``, or pointed at
``MILVUS_VALIDATION_URI`` -- see
``tests/fixtures/containers/milvus_remote.py``).

Every other integration test in this package only ever builds a
``milvusql:///path/to/file.db`` URL (the no-host, Milvus-Lite-file
branch of ``create_connect_args``) -- none of them exercise a real
host/port/credentials round trip. This module closes that gap: create
a collection, insert a row, run a plain filtered ``SELECT`` through
``AsyncConnection.execute()``, then drop the collection -- the
lifecycle is built fresh here (not shared with ``seeded_engine``/
``seeded_engine_aio``) because a real server, unlike Milvus Lite's
per-test ``tmp_path`` file, needs its own explicit create/drop instead
of just discarding a temp file."""

from __future__ import annotations

import uuid

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import BigInteger, Column, Index, MetaData, String, Table, select

pytestmark = [pytest.mark.validation, pytest.mark.sqlalchemy]


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
                milvusql_with={"metric_type": "COSINE", "M": 16, "ef_construction": 200},
            ).create(c)
        )
        await conn.commit()

    try:
        async with engine.connect() as conn:
            await conn.execute(
                items.insert(), [{"category": "book", "embedding": [0.1] * 8}]
            )
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
        assert rows == [(1, "book")]
    finally:
        async with engine.connect() as conn:
            await conn.run_sync(items.drop)
            await conn.commit()

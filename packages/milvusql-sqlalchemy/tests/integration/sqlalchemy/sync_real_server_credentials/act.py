"""Sync-side equivalent of ``async_real_server_credentials/act.py``:
proves ``milvusql://user:pass@host:port/db`` (the sync dialect's
share of the same real host/port/credentials round trip) reaches a
real Milvus server -- create, insert, plain filtered ``SELECT``,
drop."""

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


def test_credentials_host_and_port_reach_a_real_server(remote_engine):
    engine = remote_engine
    name = f"items_{uuid.uuid4().hex}"
    metadata = MetaData()
    items = Table(
        name,
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("category", String(64)),
        Column("embedding", VECTOR(8)),
        milvusql_shards=1,
        milvusql_consistency_level="Strong",
    )

    metadata.create_all(engine)
    Index(
        f"idx_{name}",
        items.c.embedding,
        milvusql_using="HNSW",
        milvusql_with={
            "metric_type": "COSINE",
            "M": 16,
            "ef_construction": 200,
        },
    ).create(engine)

    try:
        with engine.connect() as conn:
            result = conn.execute(
                items.insert(), [{"category": "book", "embedding": [0.1] * 8}]
            )
            (generated_id,) = result.inserted_primary_key
            conn.commit()
            # CREATE INDEX doesn't LOAD -- same gotcha as
            # ``seeded_engine``'s fixture docstring.
            conn.connection.dbapi_connection._client.load_collection(name)

        with engine.connect() as conn:
            rows = conn.execute(
                select(items.c.id, items.c.category).where(
                    items.c.category == "book"
                )
            ).all()
        # Milvus's real `auto_id` allocator assigns large,
        # timestamp-derived ids -- not the small sequential ones
        # Milvus Lite happened to hand out -- so this compares against
        # whatever id the insert above actually got back, not a
        # literal.
        assert rows == [(generated_id, "book")]
    finally:
        items.drop(engine)

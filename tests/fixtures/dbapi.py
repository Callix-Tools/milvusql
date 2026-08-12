"""Fixtures for the DBAPI test suite: every test gets its own Milvus
Lite instance (via ``tmp_path``), so ``pytest-randomly`` can freely
reorder tests without collection-name collisions."""

from __future__ import annotations

import pytest

import milvusql
from milvusql import aio

CREATE_ITEMS = (
    "CREATE TABLE items ("
    "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
    "embedding VECTOR(8), "
    "category VARCHAR(64)"
    ") WITH (shards=1, consistency_level='Bounded')"
)
CREATE_ITEMS_INDEX = (
    "CREATE INDEX idx_emb ON items (embedding) USING HNSW "
    "WITH (metric_type='COSINE', M=16, ef_construction=200)"
)


@pytest.fixture
def db_uri(tmp_path) -> str:
    return str(tmp_path / "milvus.db")


@pytest.fixture
def conn(db_uri):
    connection = milvusql.connect(uri=db_uri)
    yield connection
    connection.close()


@pytest.fixture
def cur(conn):
    return conn.cursor()


@pytest.fixture
def loaded_items(conn, cur):
    """A collection with an index, loaded and ready to search --
    everything CREATE INDEX needs, built through the sync client
    regardless of which client the test itself exercises (Milvus Lite
    is one on-disk server; both clients see the same collection)."""
    cur.execute(CREATE_ITEMS)
    cur.execute(CREATE_ITEMS_INDEX)
    cur.execute("LOAD TABLE items")
    return conn


@pytest.fixture
async def aconn(db_uri, loaded_items):
    connection = aio.connect(uri=db_uri)
    yield connection
    await connection.close()


@pytest.fixture
async def acur(aconn):
    return aconn.cursor()

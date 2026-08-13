"""Fixtures for the DBAPI test suite: all tests in one pytest-xdist
worker share a single real Milvus server (started via
``testcontainers``, see ``tests.fixtures.containers.milvus_server``)
and a single database dedicated to that worker (``milvus_db_name``).
Isolation across tests -- so ``pytest-randomly`` can freely reorder
them without collection-name collisions -- comes from a per-test
teardown that drops every collection in the worker's database after
each test that requests ``conn``/``aconn`` (``_milvus_worker_cleanup``)."""

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
def db_uri(milvus_uri: str) -> str:
    return milvus_uri


@pytest.fixture
def conn(db_uri, milvus_db_name, _milvus_worker_cleanup):
    connection = milvusql.connect(
        uri=db_uri,
        token="root:Milvus",  # noqa: S106 -- well-known Milvus default, not a secret
        db_name=milvus_db_name,
    )
    yield connection
    connection.close()


@pytest.fixture
def cur(conn):
    return conn.cursor()


@pytest.fixture
def loaded_items(conn, cur):
    """A collection with an index, loaded and ready to search --
    everything CREATE INDEX needs, built through the sync client
    regardless of which client the test itself exercises (both
    clients point at the same real server and database)."""
    cur.execute(CREATE_ITEMS)
    cur.execute(CREATE_ITEMS_INDEX)
    cur.execute("LOAD TABLE items")
    return conn


@pytest.fixture
async def aconn(db_uri, milvus_db_name, loaded_items):
    connection = aio.connect(
        uri=db_uri,
        token="root:Milvus",  # noqa: S106 -- well-known Milvus default, not a secret
        db_name=milvus_db_name,
    )
    yield connection
    await connection.close()


@pytest.fixture
async def acur(aconn):
    return aconn.cursor()

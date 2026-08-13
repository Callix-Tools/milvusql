"""Integration coverage exercising ``MilvusDialect_aio`` end to end
through a real ``sqlalchemy.ext.asyncio.AsyncEngine`` backed by a real
Milvus server (via ``testcontainers``): vector search, a plain filter
query, insert, delete, and reflection via ``sqlalchemy.inspect()``
(through ``AsyncConnection.run_sync()``, the only supported way to
call synchronous ``Dialect`` reflection methods against an async
engine).

Reflection specifically exercises ``dialect.py``'s ``_reflect()``
bridge: ``_raw_client()`` returns an ``AsyncMilvusClient`` under this
dialect, whose ``list_collections()``/``describe_collection()``/
``list_indexes()``/``describe_index()`` are coroutines --
``_reflect()`` runs them via ``sqlalchemy.util.await_only()`` from
inside the greenlet ``run_sync()`` spawns, the same bridge primitive
``sqlalchemy.connectors.asyncio`` itself uses for the DML/DQL path.

``seeded_engine_aio`` builds and indexes the collection through the
*sync* engine first, then reopens it async -- both clients see the
same on-disk-per-worker database, same reasoning as the root package's
own ``loaded_items`` fixture. Originally worked around a Milvus-Lite-
specific gap (``CREATE INDEX`` couldn't run through ``milvusql+aio``
against Milvus Lite's async gRPC server -- see
``ast_to_pymilvus._build_create_index``'s docstring); kept as-is
against a real server too since it hasn't been independently
reverified here that a real server's async client handles ``CREATE
INDEX`` directly, and the sync-then-reopen path is correct either
way."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import BigInteger, inspect, select

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


async def test_vector_search_returns_the_nearest_row(seeded_engine_aio):
    engine, items = seeded_engine_aio
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(items.c.id, items.c.category)
                .order_by(items.c.embedding.l2_distance([0.1] * 8))
                .limit(5)
            )
        ).all()
    assert rows == [(1, "book")]


async def test_plain_filter_select_returns_the_matching_row(
    seeded_engine_aio,
):
    engine, items = seeded_engine_aio
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(items.c.id, items.c.category).where(
                    items.c.category == "book"
                )
            )
        ).all()
    assert rows == [(1, "book")]


async def test_insert_is_visible_to_a_later_select(seeded_engine_aio):
    engine, items = seeded_engine_aio
    async with engine.connect() as conn:
        await conn.execute(
            items.insert(), [{"category": "movie", "embedding": [0.9] * 8}]
        )
        await conn.commit()
    async with engine.connect() as conn:
        rows = (await conn.execute(select(items.c.category))).all()
    assert sorted(rows) == [("book",), ("movie",)]


async def test_delete_removes_the_row(seeded_engine_aio):
    engine, items = seeded_engine_aio
    async with engine.connect() as conn:
        result = await conn.execute(
            items.delete().where(items.c.category == "book")
        )
        await conn.commit()
    assert result.rowcount == 1
    async with engine.connect() as conn:
        rows = (await conn.execute(select(items.c.id))).all()
    assert rows == []


async def test_reflected_table_names_include_the_created_collection(
    seeded_engine_aio,
):
    engine, _items = seeded_engine_aio
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda c: inspect(c).get_table_names())
    assert names == ["items"]


class TestHasTable:
    async def test_true_for_a_created_collection(self, seeded_engine_aio):
        engine, _items = seeded_engine_aio
        async with engine.connect() as conn:
            result = await conn.run_sync(
                lambda c: inspect(c).has_table("items")
            )
        assert result is True

    async def test_false_for_a_collection_that_was_never_created(
        self, seeded_engine_aio
    ):
        engine, _items = seeded_engine_aio
        async with engine.connect() as conn:
            result = await conn.run_sync(
                lambda c: inspect(c).has_table("does_not_exist")
            )
        assert result is False


async def test_reflected_columns_match_the_created_schema(seeded_engine_aio):
    engine, _items = seeded_engine_aio
    async with engine.connect() as conn:
        columns = await conn.run_sync(
            lambda c: {
                col["name"]: col for col in inspect(c).get_columns("items")
            }
        )
    assert columns["id"]["nullable"] is False
    assert columns["id"]["autoincrement"] is True
    assert isinstance(columns["id"]["type"], BigInteger)
    assert isinstance(columns["embedding"]["type"], VECTOR)
    assert columns["embedding"]["type"].dim == 8


async def test_reflected_pk_constraint_reports_the_id_column(
    seeded_engine_aio,
):
    engine, _items = seeded_engine_aio
    async with engine.connect() as conn:
        pk = await conn.run_sync(
            lambda c: inspect(c).get_pk_constraint("items")
        )
    assert pk["constrained_columns"] == ["id"]


async def test_reflected_index_reports_column_and_metric(seeded_engine_aio):
    """Same known Milvus Lite gotcha as the sync suite's equivalent
    test: ``describe_index()`` doesn't reliably preserve the literal
    name ``CREATE INDEX`` was given -- assert on the
    dialect-independent shape, not the name."""
    engine, _items = seeded_engine_aio
    async with engine.connect() as conn:
        indexes = await conn.run_sync(
            lambda c: inspect(c).get_indexes("items")
        )
    assert len(indexes) == 1
    index = indexes[0]
    assert index["column_names"] == ["embedding"]
    assert index["unique"] is False
    assert index["dialect_options"]["milvusql_using"] == "HNSW"
    assert index["dialect_options"]["milvusql_with"]["metric_type"] == "COSINE"

"""Integration coverage exercising ``MilvusDialect`` end to end through
a real ``sqlalchemy.Engine`` backed by a real Milvus server (via
``testcontainers``): ``metadata.create_all()``, ``CREATE INDEX`` +
load, insert, vector search, and reflection via ``sqlalchemy.inspect()``.
Each test rebuilds its own collection through ``seeded_engine`` (see
``tests/fixtures/engine.py``), and every matching collection is
dropped again at teardown, so ``pytest-randomly`` can freely reorder
them without collection-name collisions.

Every test in this module is a happy-path/documented-behavior
confirmation through a real engine -- including ``TestHasTable``'s
false case, which is correct documented behavior, not an error."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import BigInteger, inspect, select
from sqlalchemy.orm import Session

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


def test_legacy_query_count_flattens_its_wrapping_subquery(seeded_engine):
    """``Session.query(...).count()`` always wraps the query in a
    subquery (confirmed directly against the compiled SQL) -- a real
    regression risk from rejecting every subquery-in-FROM outright, not
    just a hypothetical one. ``_flatten_trivial_subquery`` unwraps this
    specific safe shape instead."""
    engine, items = seeded_engine
    with Session(engine) as session:
        assert session.query(items).count() == 1
        assert (
            session.query(items).filter(items.c.category == "book").count()
            == 1
        )
        assert (
            session.query(items)
            .filter(items.c.category == "nonexistent")
            .count()
            == 0
        )


def test_vector_search_returns_the_nearest_row(seeded_engine):
    engine, items = seeded_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(items.c.id, items.c.category)
            .order_by(items.c.embedding.l2_distance([0.1] * 8))
            .limit(5)
        ).all()
    assert rows == [(1, "book")]


def test_reflected_table_names_include_the_created_collection(seeded_engine):
    engine, _items = seeded_engine
    insp = inspect(engine)
    assert insp.get_table_names() == ["items"]


class TestHasTable:
    def test_true_for_a_created_collection(self, seeded_engine):
        engine, _items = seeded_engine
        assert inspect(engine).has_table("items") is True

    def test_false_for_a_collection_that_was_never_created(
        self, seeded_engine
    ):
        engine, _items = seeded_engine
        assert inspect(engine).has_table("does_not_exist") is False


def test_reflected_columns_match_the_created_schema(seeded_engine):
    engine, _items = seeded_engine
    columns = {c["name"]: c for c in inspect(engine).get_columns("items")}
    assert columns["id"]["nullable"] is False
    assert columns["id"]["autoincrement"] is True
    assert isinstance(columns["id"]["type"], BigInteger)
    assert columns["category"]["nullable"] is True
    assert columns["embedding"]["nullable"] is False
    assert isinstance(columns["embedding"]["type"], VECTOR)
    assert columns["embedding"]["type"].dim == 8


def test_reflected_pk_constraint_reports_the_id_column(seeded_engine):
    engine, _items = seeded_engine
    pk = inspect(engine).get_pk_constraint("items")
    assert pk["constrained_columns"] == ["id"]


def test_reflected_foreign_keys_are_always_empty(seeded_engine):
    """Milvus has no foreign keys -- honestly empty, not an error."""
    engine, _items = seeded_engine
    assert inspect(engine).get_foreign_keys("items") == []


def test_reflected_index_reports_column_and_metric(seeded_engine):
    """Milvus Lite's ``describe_index()`` doesn't reliably preserve the
    literal name ``CREATE INDEX`` was given (observed directly: it
    reflects back as the field name instead) -- assert on the
    dialect-independent shape, not the name."""
    engine, _items = seeded_engine
    indexes = inspect(engine).get_indexes("items")
    assert len(indexes) == 1
    index = indexes[0]
    assert index["column_names"] == ["embedding"]
    assert index["unique"] is False
    assert index["dialect_options"]["milvusql_using"] == "HNSW"
    assert index["dialect_options"]["milvusql_with"]["metric_type"] == "COSINE"


def test_get_schema_names_returns_the_default_schema(seeded_engine):
    engine, _items = seeded_engine
    assert inspect(engine).get_schema_names() == ["default"]

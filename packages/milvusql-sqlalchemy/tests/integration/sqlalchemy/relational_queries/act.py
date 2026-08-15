"""Joins, grouped aggregates and subqueries through a real
``sqlalchemy.Engine`` against a real Milvus server.

The unit tests in ``tests/unit/sqlalchemy/relational_planning`` prove
the dialect's SQL plans into the right reads. This proves the rest of
the round trip: that the reads execute, and that the rows come back
through SQLAlchemy's own result mapping with the labels the ORM asked
for."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    func,
    insert,
    select,
)

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


@pytest.fixture
def two_tables(engine):
    """``rel_items`` and ``rel_categories``, both indexed, loaded and
    seeded -- a foreign key in the ordinary sense (a plain ``BigInteger``
    column), since Milvus has no FK constraints and the dialect reports
    ``supports_foreign_keys = False``."""
    metadata = MetaData()
    categories = Table(
        "rel_categories",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=False),
        Column("title", String(64)),
        Column("v", VECTOR(2)),
        milvusql_shards=1,
        milvusql_consistency_level="Strong",
    )
    items = Table(
        "rel_items",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=False),
        Column("cat_id", BigInteger),
        Column("price", Integer),
        Column("embedding", VECTOR(8)),
        milvusql_shards=1,
        milvusql_consistency_level="Strong",
    )
    metadata.create_all(engine)
    for name, column in (
        ("idx_rel_cat_v", categories.c.v),
        ("idx_rel_item_emb", items.c.embedding),
    ):
        Index(
            name,
            column,
            milvusql_using="HNSW",
            milvusql_with={"metric_type": "COSINE"},
        ).create(engine)

    with engine.connect() as conn:
        conn.execute(
            insert(categories),
            [
                {"id": 1, "title": "book", "v": [0.1, 0.2]},
                {"id": 2, "title": "movie", "v": [0.3, 0.4]},
                {"id": 3, "title": "unused", "v": [0.5, 0.6]},
            ],
        )
        conn.execute(
            insert(items),
            [
                {"id": 1, "cat_id": 1, "price": 10, "embedding": [0.1] * 8},
                {"id": 2, "cat_id": 1, "price": 30, "embedding": [0.2] * 8},
                {"id": 3, "cat_id": 2, "price": 50, "embedding": [0.9] * 8},
                # points at no category -- the outer-join orphan
                {"id": 4, "cat_id": 99, "price": 70, "embedding": [0.5] * 8},
            ],
        )
        conn.commit()
    return engine, items, categories


def test_a_join_returns_matched_rows(two_tables):
    engine, items, categories = two_tables
    with engine.connect() as conn:
        rows = conn.execute(
            select(items.c.id, categories.c.title)
            .join(categories, items.c.cat_id == categories.c.id)
            .order_by(items.c.id)
        ).all()
    assert [tuple(row) for row in rows] == [
        (1, "book"),
        (2, "book"),
        (3, "movie"),
    ]


def test_an_outer_join_keeps_the_orphan(two_tables):
    engine, items, categories = two_tables
    with engine.connect() as conn:
        rows = conn.execute(
            select(items.c.id, categories.c.title)
            .outerjoin(categories, items.c.cat_id == categories.c.id)
            .order_by(items.c.id)
        ).all()
    assert [tuple(row) for row in rows][-1] == (4, None)


def test_a_filter_on_the_joined_table_is_applied(two_tables):
    engine, items, categories = two_tables
    with engine.connect() as conn:
        rows = conn.execute(
            select(items.c.id)
            .join(categories, items.c.cat_id == categories.c.id)
            .where(categories.c.title == "movie")
        ).all()
    assert [tuple(row) for row in rows] == [(3,)]


def test_a_grouped_aggregate_labels_its_columns(two_tables):
    engine, items, _categories = two_tables
    with engine.connect() as conn:
        result = conn.execute(
            select(
                items.c.cat_id,
                func.count().label("n"),
                func.sum(items.c.price).label("total"),
            )
            .group_by(items.c.cat_id)
            .order_by(items.c.cat_id)
        )
        assert list(result.keys()) == ["cat_id", "n", "total"]
        assert [tuple(row) for row in result.all()] == [
            (1, 2, 40),
            (2, 1, 50),
            (99, 1, 70),
        ]


def test_an_in_subquery_over_the_other_collection(two_tables):
    engine, items, categories = two_tables
    with engine.connect() as conn:
        rows = conn.execute(
            select(items.c.id)
            .where(
                items.c.cat_id.in_(
                    select(categories.c.id).where(
                        categories.c.title == "book"
                    )
                )
            )
            .order_by(items.c.id)
        ).all()
    assert [tuple(row) for row in rows] == [(1,), (2,)]

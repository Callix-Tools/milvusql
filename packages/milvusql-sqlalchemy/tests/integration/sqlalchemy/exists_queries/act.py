"""Integration coverage for correlated ``[NOT] EXISTS`` through the
standard SQLAlchemy surface against a real Milvus server -- the shape
``relationship(...).any()``/``.has()`` compile to, spelled with Core's
``exists()`` (identical emitted SQL)."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Index,
    MetaData,
    Table,
    exists,
    insert,
    not_,
    select,
    true,
)

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


@pytest.fixture
def parent_child(engine):
    metadata = MetaData()
    parents = Table(
        "sa_parents",
        metadata,
        # autoincrement=False: explicit ids below -- the default would
        # render AUTO_INCREMENT, and Milvus's auto_id primary keys
        # reject caller-provided values outright.
        Column("id", BigInteger, primary_key=True, autoincrement=False),
        Column("v", VECTOR(2)),
        milvusql_consistency_level="Strong",
    )
    children = Table(
        "sa_children",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=False),
        Column("parent_id", BigInteger),
        Column("active", Boolean),
        Column("v", VECTOR(2)),
        milvusql_consistency_level="Strong",
    )
    for table in (parents, children):
        Index(
            f"idx_{table.name}_v",
            table.c.v,
            milvusql_using="FLAT",
            milvusql_with={"metric_type": "L2"},
        )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(parents),
            [{"id": i, "v": [0.0, 0.0]} for i in (1, 2, 3)],
        )
        connection.execute(
            insert(children),
            [
                {"id": 10, "parent_id": 1, "active": True, "v": [0.0, 0.0]},
                {"id": 11, "parent_id": 2, "active": False, "v": [0.0, 0.0]},
            ],
        )
    return parents, children


def test_exists_keeps_only_matching_parents(parent_child, engine):
    parents, children = parent_child
    with engine.connect() as connection:
        rows = connection.execute(
            select(parents.c.id).where(
                exists(
                    select(1).where(
                        parents.c.id == children.c.parent_id,
                        # `== true()`, not `.is_(True)`: Milvus's filter
                        # DSL has `IS NULL` only, no `IS TRUE`.
                        children.c.active == true(),
                    )
                )
            )
        ).all()
    assert rows == [(1,)]


def test_not_exists_is_the_anti_join(parent_child, engine):
    parents, children = parent_child
    with engine.connect() as connection:
        rows = connection.execute(
            select(parents.c.id).where(
                not_(
                    exists(
                        select(1).where(parents.c.id == children.c.parent_id)
                    )
                )
            )
        ).all()
    assert rows == [(3,)]

"""Regression coverage for a bug found while exercising every documented
``milvusql-sqlalchemy`` SQLAlchemy feature end to end: the ORM
(``sqlalchemy.orm.Session``) could not insert a row through a mapped class
with an autoincrement/``auto_id`` primary key -- ``session.commit()``
always raised ``sqlalchemy.orm.exc.FlushError: ... has a NULL identity
key``.

Root cause, confirmed directly: ``MilvusDialect.postfetch_lastrowid`` was
set to ``False`` in ``dialect.py``, so SQLAlchemy never read
``cursor.lastrowid`` back after an ``INSERT`` -- even though
``milvusql.dbapi.Cursor.execute`` *does* correctly set it (verified
directly: ``cursor.lastrowid`` holds Milvus's server-assigned ``auto_id``
value after every insert, wired all the way from
``ast_to_pymilvus._insert_result``). With ``postfetch_lastrowid`` off,
``Core``-level ``result.inserted_primary_key`` was also always
``(None,)`` for the same reason -- only the raw DBAPI cursor's
``.lastrowid`` attribute (not exposed through ``Connection.execute()``'s
``Result``) actually carried the generated id. The ORM's unit-of-work
needs the generated primary key back to register the new object as
persistent, so every ``Session.add(obj)`` + ``session.commit()`` for a
model with an autoincrement primary key failed.

Fixed by flipping ``postfetch_lastrowid`` back to ``True`` (the base
``DefaultDialect``'s own default) -- see that attribute's docstring in
``dialect.py``. Plain ``Core`` inserts
(``conn.execute(insert(table), {...})``, used throughout the rest of this
test suite) were never affected -- they don't ask for the generated id
back either way.
"""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import BigInteger, Index, String, insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


class _Base(DeclarativeBase):
    pass


class _Item(_Base):
    __tablename__ = "orm_items"
    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    category: Mapped[str] = mapped_column(String(64))
    embedding = mapped_column(VECTOR(4))


def _create_and_load(engine):
    """``LOAD`` requires an index on every vector field first -- Milvus
    Lite tolerated loading an unindexed collection (confirmed
    directly: these two tests used to skip index creation entirely and
    passed against Lite regardless), but a real server rejects it with
    ``index not found``. Milvus Lite's own leniency there was the gap,
    not a real Milvus capability these tests should keep assuming."""
    _Base.metadata.create_all(engine)
    Index(
        "idx_orm_items_embedding",
        _Item.embedding,
        milvusql_using="HNSW",
        milvusql_with={"metric_type": "COSINE", "M": 16, "ef_construction": 200},
    ).create(engine)
    with engine.connect() as conn:
        dbapi_connection = conn.connection.dbapi_connection
        assert dbapi_connection is not None  # nosec B101
        dbapi_connection._client.load_collection("orm_items")


def test_session_add_and_commit_assigns_the_generated_primary_key(engine):
    _create_and_load(engine)

    with Session(engine) as session:
        obj = _Item(category="book", embedding=[0.1] * 4)
        session.add(obj)
        session.commit()
        assert obj.id is not None


def test_core_insert_reports_the_generated_primary_key(engine):
    _create_and_load(engine)

    with engine.connect() as conn:
        result = conn.execute(
            insert(_Item),
            {"category": "movie", "embedding": [0.2] * 4},
        )
        assert result.inserted_primary_key is not None
        assert result.inserted_primary_key[0] is not None

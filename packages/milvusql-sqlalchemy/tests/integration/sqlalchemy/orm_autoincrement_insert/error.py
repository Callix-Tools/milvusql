"""Regression coverage for a bug found while exercising every documented
``milvusql-sqlalchemy`` SQLAlchemy feature end to end: the ORM
(``sqlalchemy.orm.Session``) cannot insert a row through a mapped class
with an autoincrement/``auto_id`` primary key -- ``session.commit()``
always raises ``sqlalchemy.orm.exc.FlushError: ... has a NULL identity
key``.

Root cause, confirmed directly: ``MilvusDialect.postfetch_lastrowid`` is
set to ``False`` in ``dialect.py``, so SQLAlchemy never reads
``cursor.lastrowid`` back after an ``INSERT`` -- even though
``milvusql.dbapi.Cursor.execute`` *does* correctly set it (verified
directly: ``cursor.lastrowid`` holds Milvus's server-assigned ``auto_id``
value after every insert, wired all the way from
``ast_to_pymilvus._insert_result``). Because ``postfetch_lastrowid`` is
off, ``Core``-level ``result.inserted_primary_key`` is also always
``(None,)`` for the same reason -- only the raw DBAPI cursor's
``.lastrowid`` attribute (not exposed through ``Connection.execute()``'s
``Result``) actually carries the generated id. The ORM's unit-of-work
needs the generated primary key back to register the new object as
persistent, so every ``Session.add(obj)`` + ``session.commit()`` for a
model with an autoincrement primary key fails.

Plain ``Core`` inserts (``conn.execute(insert(table), {...})``, as used
throughout the rest of this test suite) are unaffected -- they never ask
for the generated id back.
"""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import BigInteger, String
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MilvusDialect.postfetch_lastrowid = False means SQLAlchemy never "
        "reads back the auto_id cursor.lastrowid the DBAPI already "
        "provides, so the ORM unit-of-work can't learn a new object's "
        "primary key. See this file's module docstring."
    ),
)
def test_session_add_and_commit_assigns_the_generated_primary_key(engine):
    _Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.connection.dbapi_connection._client.load_collection("orm_items")

    with Session(engine) as session:
        obj = _Item(category="book", embedding=[0.1] * 4)
        session.add(obj)
        session.commit()
        assert obj.id is not None

"""Regression coverage for a bug found while exercising every documented
``milvusql-sqlalchemy`` SQLAlchemy feature end to end: the isolation-level
extension point (``dialect.py``'s D6 -- ``get_isolation_level``/
``set_isolation_level`` repurposed to carry Milvus's *consistency level*)
was unusable through SQLAlchemy's own public API.

Root cause, confirmed directly: ``sqlalchemy.engine.default.DefaultDialect
._assert_and_set_isolation_level`` -- the method SQLAlchemy actually calls
for both ``create_engine(isolation_level=...)`` and
``Connection.execution_options(isolation_level=...)`` -- unconditionally
does ``level.replace("_", " ").upper()`` before handing the value to
``dialect.set_isolation_level()``. That's correct for the standard SQL
isolation levels this hook was designed for ("READ COMMITTED", ...), but
Milvus's own consistency-level names are title-case ("Bounded", "Strong",
"Session", "Eventually", "Customized") and ``pymilvus``'s
``get_consistency_level()`` does an exact, case-sensitive
``ConsistencyLevel.Value(...)`` lookup -- so the uppercased value
("BOUNDED", "STRONG", ...) always failed with
``pymilvus.exceptions.InvalidConsistencyLevel``, on every single query.

It got worse than "the feature doesn't work": SQLAlchemy's connection
pool also uses this same machinery to *reset* isolation level on
checkin (``reset_isolation_level``, registered as a
``finalize_callback`` the moment ``execution_options(isolation_level=...)``
is used once). That reset also went through the uppercasing method, using
``dialect.default_isolation_level`` (computed once, from this dialect's
own "Bounded" fallback) -- so touching ``isolation_level`` *once*, on any
connection pulled from the pool, permanently poisoned that pooled DBAPI
connection: every later, otherwise unrelated query issued on it also
started failing the same way, for the lifetime of the process.

Fixed by ``MilvusDialect._assert_and_set_isolation_level`` overriding the
base method entirely instead of fighting its uppercasing after the fact --
see that method's own docstring in ``dialect.py``.
"""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import (
    BigInteger,
    Column,
    MetaData,
    Table,
    create_engine,
    insert,
    select,
)

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


def test_engine_level_isolation_level_does_not_break_every_query(db_uri):
    engine = create_engine(db_uri, isolation_level="Bounded")
    try:
        metadata = MetaData()
        items = Table(
            "items",
            metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("embedding", VECTOR(4)),
        )
        metadata.create_all(engine)
        with engine.connect() as conn:
            dbapi_connection = conn.connection.dbapi_connection
            assert dbapi_connection is not None  # nosec B101
            dbapi_connection._client.load_collection("items")
            conn.execute(insert(items), [{"embedding": [0.1] * 4}])
            conn.commit()
        with engine.connect() as conn:
            rows = conn.execute(select(items.c.id)).all()
        assert rows == [(1,)]
    finally:
        engine.dispose()


def test_execution_options_isolation_level_does_not_poison_the_pool(
    seeded_engine,
):
    engine, items = seeded_engine

    with engine.connect() as raw_conn:
        conn = raw_conn.execution_options(isolation_level="Strong")
        rows = conn.execute(select(items.c.id)).all()
        assert rows == [(1,)]

    # A later, unrelated connection pulled from the same pool must not
    # be affected by the isolation_level use above.
    with engine.connect() as conn:
        rows = conn.execute(select(items.c.id)).all()
        assert rows == [(1,)]

"""Coverage for two DDL gaps found while exercising every documented
``milvusql-sqlalchemy`` SQLAlchemy feature end to end, both now fixed in
``milvusql`` core's ``ast_to_pymilvus`` dispatch/type-mapping:

* ``DROP TABLE`` -- and therefore the everyday ``Table.drop(engine)``/
  ``MetaData.drop_all(engine)`` calls that compile to it -- had no
  builder at all in ``ast_to_pymilvus._BUILDERS``/``build_call``
  (confirmed directly: unlike ``ALTER TABLE``, which raises its own
  dedicated, explicit "not implemented yet" message, a parsed ``DROP
  TABLE`` fell through to the generic "unsupported statement: Drop").
  Fixed by ``build_call``'s new ``exp.Drop`` branch (mirroring the
  existing ``exp.Create`` one) plus ``_build_drop_table``, calling
  ``MilvusClient.drop_collection``.

* ``SPARSEVEC`` -- exported from ``milvusql_sqlalchemy``'s top-level
  ``__all__``, with a working comparator (``max_inner_product``) and
  ``CREATE TABLE`` column-spec rendering ("SPARSEVEC") -- could never
  actually create a collection: ``_map_datatype`` had no mapping for it.
  Fixed by matching ``SPARSEVEC``'s parsed shape (a generic
  ``exp.DataType.Type.USERDEFINED`` carrying "SPARSEVEC" as free-text
  ``kind`` -- confirmed directly, sqlglot has no dedicated enum member
  for it the way it does for ``VECTOR``) to ``DataType
  .SPARSE_FLOAT_VECTOR``, and mapping that type back to ``SPARSEVEC()``
  on reflection (``milvusql_sqlalchemy.reflection``).
"""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR
from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    MetaData,
    Table,
    insert,
    inspect,
    select,
)

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


def test_drop_table_removes_the_collection(engine):
    metadata = MetaData()
    items = Table(
        "dropme",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("embedding", VECTOR(4)),
    )
    metadata.create_all(engine)
    items.drop(engine)

    assert inspect(engine).has_table("dropme") is False


def test_sparsevec_column_can_be_created_inserted_and_searched(engine):
    metadata = MetaData()
    docs = Table(
        "docs",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("sparse", SPARSEVEC()),
    )
    metadata.create_all(engine)
    Index(
        "idx_sparse",
        docs.c.sparse,
        milvusql_using="SPARSE_INVERTED_INDEX",
        milvusql_with={"metric_type": "IP"},
    ).create(engine)

    with engine.connect() as conn:
        dbapi_connection = conn.connection.dbapi_connection
        assert dbapi_connection is not None  # nosec B101
        dbapi_connection._client.load_collection("docs")

        conn.execute(insert(docs), [{"sparse": {0: 0.5, 5: 0.2}}])
        conn.commit()

        rows = conn.execute(
            select(docs.c.id)
            .order_by(docs.c.sparse.max_inner_product({0: 0.5}))
            .limit(5)
        ).all()
        assert rows == [(1,)]


def test_sparsevec_column_reflects_back_as_sparsevec(engine):
    metadata = MetaData()
    Table(
        "docs2",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("sparse", SPARSEVEC()),
    )
    metadata.create_all(engine)

    columns = {c["name"]: c for c in inspect(engine).get_columns("docs2")}
    assert isinstance(columns["sparse"]["type"], SPARSEVEC)

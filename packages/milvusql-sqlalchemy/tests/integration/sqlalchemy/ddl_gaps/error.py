"""Coverage for two undocumented DDL gaps found while exercising every
documented ``milvusql-sqlalchemy`` SQLAlchemy feature end to end. Both
raise ``NotSupportedError`` consistently (no silent wrong behavior), so
these are not regressions the way the other two new test modules in this
directory are -- just gaps worth pinning down and calling out, since
neither the root README nor the ``milvusql-sqlalchemy`` README mentions
them:

* ``DROP TABLE`` -- and therefore the everyday ``Table.drop(engine)``/
  ``MetaData.drop_all(engine)`` calls that compile to it -- has no
  builder at all in ``milvusql``'s ``ast_to_pymilvus._BUILDERS`` dispatch
  table (confirmed directly: unlike ``ALTER TABLE``, which raises its own
  dedicated, explicit "not implemented yet" message, a parsed ``DROP
  TABLE`` falls through to the generic "unsupported statement: Drop").
  ``compiler.py``'s own module docstring says "CREATE TABLE/DROP TABLE
  parse for free" -- true only for *parsing* (``sqlglot`` accepts the
  grammar), not execution.

* ``SPARSEVEC`` -- exported from ``milvusql_sqlalchemy``'s top-level
  ``__all__``, with a working comparator (``max_inner_product``) and
  ``CREATE TABLE`` column-spec rendering ("SPARSEVEC") -- can never
  actually create a collection: ``milvusql``'s own
  ``ast_to_pymilvus._map_datatype`` has no mapping for it (its own
  docstring calls sparse/binary vector spellings "a real gap, not a
  silent guess"), so ``CREATE TABLE ... sparse SPARSEVEC`` always raises
  ``NotSupportedError: unsupported column type: SPARSEVEC``. The type is
  usable for building/compiling SQL text today, but not for ever running
  it against a real collection through this dialect.
"""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR
from sqlalchemy import BigInteger, Column, MetaData, Table
from sqlalchemy.exc import NotSupportedError

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


def test_drop_table_is_not_supported(engine):
    metadata = MetaData()
    items = Table(
        "dropme",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("embedding", VECTOR(4)),
    )
    metadata.create_all(engine)
    with pytest.raises(NotSupportedError, match="Drop"):
        items.drop(engine)


def test_sparsevec_column_cannot_be_created(engine):
    metadata = MetaData()
    Table(
        "docs",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("sparse", SPARSEVEC()),
    )
    with pytest.raises(NotSupportedError, match="SPARSEVEC"):
        metadata.create_all(engine)

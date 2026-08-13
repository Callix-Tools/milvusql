"""Milvus refuses ``CREATE TABLE``/``create_collection()`` outright for
a schema with zero vector fields (confirmed directly, the same real
restriction ``milvusql_django.schema``'s own ``PAD_VECTOR_FIELD``
docstring documents) -- a hard problem for anything that creates a
table with no vector column of its own, chiefly Alembic's
``alembic_version`` bookkeeping table (``version_num VARCHAR(32)
PRIMARY KEY``, no vector field anywhere) and any other tool that
issues a bare ``CREATE TABLE`` through this dialect the same way.

Unlike ``milvusql_django``, there is no single schema-editor choke
point every table creation flows through here -- ``Table.create()``,
``metadata.create_all()``, and Alembic's own internal
``MigrationContext._version.create()`` all end up compiling DDL
through ``MilvusDDLCompiler`` (``compiler.py``, dialect-registered) and
then executing it through a plain ``Connection``, so the fix lives at
that shared layer instead: ``compiler.py``'s ``visit_create_table``
appends a hidden pad column to the *rendered SQL text* whenever a
table has no real ``VECTOR``/``SPARSEVEC`` column, and ``dialect.py``
registers a ``sqlalchemy.event`` ``after_create`` listener (global, not
per-``Table``, precisely so it also catches tables Alembic creates
without ever importing this package's own ``Table``/``MetaData``
objects) that indexes and loads that hidden column right after -- the
same reason Milvus needs an index and a ``LOAD`` before *any* query
succeeds against a real server, not just vector search, confirmed
directly against ``milvusql_django``'s identical fix for
``django_migrations``."""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from sqlalchemy.sql.schema import Table

#: Not user-facing -- never reflected back by ``get_table_names()``'s
#: own column listing calls, and named with a leading underscore/
#: package prefix precisely so it can't collide with a real column
#: name a caller picks.
PAD_VECTOR_FIELD = "_milvusql_pad_vector"
#: Milvus's minimum vector dimension is 2 (confirmed directly against
#: a real server -- Milvus Lite didn't enforce this), matching
#: ``milvusql_django``'s own ``PAD_VECTOR_VALUE``.
PAD_VECTOR_DIM = 2


def table_needs_pad_vector(table: Table) -> bool:
    """Whether ``table`` has no real vector column of its own -- local
    import of the vector types avoids a compiler.py <-> types.py
    import cycle (``types.py`` has no reason to import this module)."""
    from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR  # noqa: PLC0415

    return not any(
        isinstance(column.type, (VECTOR, SPARSEVEC))
        for column in table.columns
    )


__all__ = ["PAD_VECTOR_DIM", "PAD_VECTOR_FIELD", "table_needs_pad_vector"]

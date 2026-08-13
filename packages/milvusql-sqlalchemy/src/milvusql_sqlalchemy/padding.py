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
then executing it through a plain ``Connection``. The fix is a real
``Column`` -- not just SQL-text splicing -- appended directly onto the
live ``Table`` object the moment either compiler first sees it needs
one: ``get_column_specification`` (inherited, unmodified) then renders
it into ``CREATE TABLE`` exactly like any other column, and because it
carries a client-side ``default``, SQLAlchemy's own DML-compilation
machinery (``Insert._compile_state_factory``) automatically includes
it -- and its value -- in *every* later ``INSERT`` against this table
without this module doing anything further, the same way any ordinary
column-level ``default=`` works.

That "every later ``INSERT``" is why this can't live only in
``visit_create_table``: ``Table.create(checkfirst=True)`` (what
``MigrationContext.configure()`` actually calls) skips the ``CREATE
TABLE`` compile entirely when the table already exists server-side --
the overwhelmingly common case after the very first ``alembic
upgrade`` ever run against a given database, confirmed directly (a
second run's version-stamp ``INSERT`` reused the *same* Python
``Table`` object from a fresh process that never compiled ``CREATE
TABLE`` for it at all). ``MilvusSQLCompiler.visit_insert`` calls this
too, for exactly that reason -- ``ensure_pad_vector_column`` is cheap
and idempotent to call from both places (a table that already has the
column, from either path or from being reflected with it already
present server-side, is a no-op: ``table_needs_pad_vector`` sees the
pad column's own ``VECTOR`` type and returns ``False``)."""

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
#: A fresh list per call (the ``default=`` callable below), not a
#: single shared mutable object every insert would otherwise alias.
PAD_VECTOR_VALUE: list[float] = [0.0] * PAD_VECTOR_DIM


def table_needs_pad_vector(table: Table) -> bool:
    """Whether ``table`` has no real vector column of its own -- local
    import of the vector types avoids a compiler.py <-> types.py
    import cycle (``types.py`` has no reason to import this module)."""
    from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR  # noqa: PLC0415

    return not any(
        isinstance(column.type, (VECTOR, SPARSEVEC))
        for column in table.columns
    )


def ensure_pad_vector_column(table: Table) -> t.Any:
    """Idempotent -- see the module docstring for why this has to be
    callable from both DDL- and DML-compile time, safely, as many
    times as either fires. Returns the newly appended ``Column``, or
    ``None`` if the table already had one (from an earlier call, or
    from being reflected with it already present server-side) --
    ``MilvusDDLCompiler.visit_create_table`` needs that ``Column``
    back to splice a matching ``CreateColumn`` into its own
    already-snapshotted ``create.columns`` list (confirmed directly:
    ``CreateTable.__init__`` copies ``element.columns`` into
    ``self.columns`` at *construction* time, before this function ever
    runs at compile time -- appending to the live ``Table`` alone,
    after that snapshot was taken, would otherwise never make it into
    the rendered ``CREATE TABLE`` text)."""
    if not table_needs_pad_vector(table):
        return None
    from sqlalchemy.sql.schema import Column  # noqa: PLC0415

    from milvusql_sqlalchemy.types import VECTOR  # noqa: PLC0415

    column = Column(
        PAD_VECTOR_FIELD,
        VECTOR(PAD_VECTOR_DIM),
        default=lambda: list(PAD_VECTOR_VALUE),
    )
    table.append_column(column)
    return column


__all__ = [
    "PAD_VECTOR_DIM",
    "PAD_VECTOR_FIELD",
    "PAD_VECTOR_VALUE",
    "ensure_pad_vector_column",
    "table_needs_pad_vector",
]

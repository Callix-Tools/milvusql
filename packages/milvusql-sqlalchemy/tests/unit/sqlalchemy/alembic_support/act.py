"""Unit coverage for the two pieces Alembic support actually needs
from this dialect, both exercisable with zero network I/O:
``alembic_impl.py``'s ``DefaultImpl`` registration (``dialect.py``
imports it, which is what registers ``"milvusql"`` with Alembic --
without it, ``MigrationContext.configure()`` raises a bare
``KeyError`` looking the dialect up by name, confirmed directly), and
``padding.py``'s ``table_needs_pad_vector`` (the predicate
``compiler.py``'s ``visit_create_table`` and ``dialect.py``'s
``after_create`` event listener both key off of)."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.dialect import MilvusDialect
from milvusql_sqlalchemy.padding import (
    PAD_VECTOR_FIELD,
    ensure_pad_vector_column,
    table_needs_pad_vector,
)
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR
from sqlalchemy import BigInteger, Column, MetaData, String, Table, insert
from sqlalchemy.schema import CreateTable

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


@pytest.fixture
def dialect() -> MilvusDialect:
    return MilvusDialect()


def test_importing_the_dialect_registers_it_with_alembic():
    # Imported here, not at module top level: the whole point is to
    # prove that *this specific import* (`dialect.py`'s own, which
    # every `create_engine("milvusql://...")` call already triggers
    # via SQLAlchemy's dialect-loading machinery) is what performs the
    # registration -- importing it anywhere else in this test module
    # first would make that unverifiable.
    import milvusql_sqlalchemy.dialect  # noqa: F401, PLC0415
    from alembic.ddl.impl import _impls  # noqa: PLC0415

    impl_cls = _impls["milvusql"]
    assert impl_cls.__name__ == "MilvusImpl"
    # Milvus has no multi-statement rollback (D7) -- see
    # `alembic_impl.MilvusImpl`'s own docstring for why this matters.
    assert impl_cls.transactional_ddl is False


class TestTableNeedsPadVector:
    def test_true_for_a_table_with_no_vector_column(self):
        metadata = MetaData()
        alembic_version = Table(
            "alembic_version",
            metadata,
            Column("version_num", String(32), primary_key=True),
        )
        assert table_needs_pad_vector(alembic_version) is True

    def test_false_for_a_table_with_a_dense_vector_column(self):
        metadata = MetaData()
        items = Table(
            "items",
            metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("embedding", VECTOR(8)),
        )
        assert table_needs_pad_vector(items) is False

    def test_false_for_a_table_with_only_a_sparse_vector_column(self):
        metadata = MetaData()
        docs = Table(
            "docs",
            metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("sparse", SPARSEVEC()),
        )
        assert table_needs_pad_vector(docs) is False


class TestPadVectorInsertion:
    def test_create_table_includes_the_pad_column(self, dialect):
        """Regression coverage for the actual real-server bug this
        whole module exists to fix: an earlier version of this
        splicing appended the pad column onto the live ``Table``
        object *after* ``CreateTable.__init__`` had already
        snapshotted ``element.columns`` into its own ``create.columns``
        list -- the column reached the ``Table`` but never the
        rendered SQL text."""
        metadata = MetaData()
        alembic_version = Table(
            "alembic_version",
            metadata,
            Column("version_num", String(32), primary_key=True),
        )
        sql = str(CreateTable(alembic_version).compile(dialect=dialect))
        assert f"{PAD_VECTOR_FIELD} VECTOR(2)" in sql

    def test_insert_includes_the_pad_column_even_without_create_table(
        self, dialect
    ):
        """The other real-server bug: ``Table.create(checkfirst=True)``
        (what ``alembic.runtime.migration.MigrationContext.configure()``
        actually calls) skips ``CREATE TABLE`` entirely once the table
        already exists server-side -- the common case after the first
        ``alembic upgrade`` ever run. A later ``INSERT`` against that
        same (fresh-process) ``Table`` object, with no ``CREATE TABLE``
        ever compiled for it in this process, still needs the pad
        column -- confirmed here without even compiling ``CREATE
        TABLE`` first, unlike the previous test."""
        metadata = MetaData()
        alembic_version = Table(
            "alembic_version",
            metadata,
            Column("version_num", String(32), primary_key=True),
        )
        stmt = insert(alembic_version).values(version_num="0001")
        sql = str(stmt.compile(dialect=dialect))
        assert PAD_VECTOR_FIELD in sql

    def test_is_idempotent_across_repeated_calls(self):
        metadata = MetaData()
        alembic_version = Table(
            "alembic_version",
            metadata,
            Column("version_num", String(32), primary_key=True),
        )
        first = ensure_pad_vector_column(alembic_version)
        second = ensure_pad_vector_column(alembic_version)
        assert first is not None
        assert second is None
        assert list(alembic_version.columns.keys()).count(PAD_VECTOR_FIELD) == 1

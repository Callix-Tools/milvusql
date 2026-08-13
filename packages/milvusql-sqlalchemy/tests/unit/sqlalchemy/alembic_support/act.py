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
from milvusql_sqlalchemy.padding import table_needs_pad_vector
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR
from sqlalchemy import BigInteger, Column, MetaData, String, Table

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


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

"""Unit coverage for the one piece of real Alembic support this
dialect ships: ``alembic_impl.py``'s ``DefaultImpl`` registration.
Exercisable with zero network I/O -- ``dialect.py`` imports
``alembic_impl``, which is what registers ``"milvusql"`` with Alembic;
without it, ``MigrationContext.configure()`` raises a bare ``KeyError``
looking the dialect up by name, confirmed directly.

Alembic itself still can't fully drive a migration against this
backend: its own ``alembic_version`` bookkeeping table has no vector
column, and Milvus refuses ``CREATE TABLE`` outright for a schema with
zero vector fields (a real, confirmed, documented limitation, not
something this package works around -- see
``ast_to_pymilvus._build_create_table``'s own explicit
``NotSupportedError`` for it, covered in ``tests/unit/translate/
create_table/error.py``). This registration is still worth having on
its own: it's what turns Alembic's *own* unhelpful ``KeyError`` into
this backend's clear, explicit error instead, the moment
``MigrationContext.configure()`` gets far enough to try creating that
table."""

from __future__ import annotations

import pytest

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

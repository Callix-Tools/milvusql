"""Unit coverage for ``MilvusDialect_aio``'s own class-level surface --
everything the async dialect changes or adds relative to the sync
``MilvusDialect`` it inherits from: ``driver``/``is_async``/
``supports_statement_cache``/``poolclass``, ``import_dbapi()``, and the
``_unwrap()`` helper that lets ``get_isolation_level``/
``set_isolation_level`` see through the ``AsyncAdapt_dbapi_connection``
wrapper SQLAlchemy's async engine hands them instead of a raw
``milvusql.aio.AsyncConnection``. Reflection (DDL rendering, columns,
indexes, ...) is inherited unchanged from ``MilvusDialect`` and already
covered by its own unit/integration suites -- not repeated here."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy import asyncio_dbapi
from milvusql_sqlalchemy.dialect import MilvusDialect, MilvusDialect_aio
from sqlalchemy import pool

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


@pytest.fixture
def dialect() -> MilvusDialect_aio:
    return MilvusDialect_aio()


class TestDriverIdentity:
    def test_driver_is_aio(self, dialect):
        assert dialect.driver == "aio"

    def test_is_async_is_true(self, dialect):
        assert dialect.is_async is True

    def test_name_is_inherited_from_the_sync_dialect(self, dialect):
        assert dialect.name == "milvusql"


def test_supports_statement_cache_is_true():
    """SQLAlchemy re-checks this per dialect *class*, not just via
    inheritance -- confirmed directly: omitting it (even though
    ``MilvusDialect`` already sets it ``True``) makes
    ``create_async_engine`` warn "will not make use of SQL compilation
    caching" at import time."""
    assert MilvusDialect_aio.supports_statement_cache is True


def test_poolclass_is_the_async_adapted_queue_pool(dialect):
    """``DefaultDialect.get_pool_class`` just reads this attribute,
    falling back to plain ``QueuePool`` if unset -- it does not branch
    on ``is_async`` itself, so every async dialect sets this
    explicitly. Without it, ``create_async_engine`` raises "Pool class
    QueuePool cannot be used with asyncio engine" (confirmed
    directly)."""
    assert dialect.poolclass is pool.AsyncAdaptedQueuePool


def test_import_dbapi_returns_the_async_dbapi_module_singleton():
    assert MilvusDialect_aio.import_dbapi() is asyncio_dbapi.dbapi


def test_is_a_subclass_of_the_sync_dialect():
    """DDL rendering, reflection, ``construct_arguments``, and the
    consistency-level fallback are all inherited unchanged -- only the
    driver name and DBAPI module differ."""
    assert issubclass(MilvusDialect_aio, MilvusDialect)


class _WrappedConnection:
    """Stands in for the ``AsyncAdapt_dbapi_connection`` SQLAlchemy's
    async engine actually hands ``get_isolation_level``/
    ``set_isolation_level`` -- the real ``milvusql.aio.AsyncConnection``
    sits behind its ``.driver_connection`` attribute, not exposed
    directly."""

    def __init__(self):
        self.driver_connection = _StubInnerConnection()


class _StubInnerConnection:
    consistency_level = None


class TestUnwrappingIsolationLevel:
    """``_unwrap()`` is private, but every dialect method that touches
    connection state routes through it -- exercised here via the
    public ``get_isolation_level``/``set_isolation_level`` API against
    a wrapped connection, the shape the async engine actually passes."""

    def test_get_defaults_to_bounded_through_the_wrapper(self, dialect):
        assert dialect.get_isolation_level(_WrappedConnection()) == "Bounded"

    def test_set_then_get_round_trips_through_the_wrapper(self, dialect):
        wrapped = _WrappedConnection()
        dialect.set_isolation_level(wrapped, "Strong")
        assert wrapped.driver_connection.consistency_level == "Strong"
        assert dialect.get_isolation_level(wrapped) == "Strong"

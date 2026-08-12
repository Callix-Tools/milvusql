"""Unit coverage for ``asyncio_dbapi._AsyncAdaptMilvusqlDBAPI`` -- the
DBAPI-module stand-in ``MilvusDialect_aio.import_dbapi()`` returns.
Entirely offline: ``connect()`` builds a ``milvusql.aio.AsyncConnection``
directly (no I/O at construction time, same as the sync ``MilvusClient``
-- see ``milvusql.aio.connect``'s own docstring), so this never touches
a real Milvus instance."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy import asyncio_dbapi
from sqlalchemy.connectors.asyncio import AsyncAdapt_dbapi_connection

import milvusql.aio
from milvusql.dbapi import errors

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


def test_paramstyle_is_named():
    assert asyncio_dbapi.dbapi.paramstyle == "named"


@pytest.mark.parametrize(
    "name",
    [
        "Warning",
        "Error",
        "InterfaceError",
        "DatabaseError",
        "DataError",
        "OperationalError",
        "IntegrityError",
        "InternalError",
        "ProgrammingError",
        "NotSupportedError",
    ],
)
def test_exception_attribute_is_the_same_object_as_milvusql_dbapi_errors(
    name,
):
    """SQLAlchemy does ``isinstance`` checks against ``self.dbapi.Error``
    et al -- these have to be the *same* classes the sync dialect's
    error translation already produces, not lookalikes, so a caught
    exception's type matches regardless of which engine (sync or
    async) raised it."""
    assert getattr(asyncio_dbapi.dbapi, name) is getattr(errors, name)


def test_connect_returns_an_adapter_wrapping_a_real_async_connection(
    tmp_path,
):
    """A Milvus Lite target still needs an existing parent directory --
    ``AsyncMilvusClient.__init__`` validates that eagerly (confirmed
    directly: a nonexistent directory raises immediately) -- but opening
    the local file itself is the *only* I/O construction does, matching
    the module docstring's "no ``await_only()`` dance at connect time"
    claim."""
    conn = asyncio_dbapi.dbapi.connect(uri=str(tmp_path / "milvus.db"))
    assert isinstance(conn, AsyncAdapt_dbapi_connection)
    assert isinstance(conn.driver_connection, milvusql.aio.AsyncConnection)


def test_dbapi_module_singleton_is_an_instance_of_the_adapter_class():
    assert isinstance(
        asyncio_dbapi.dbapi, asyncio_dbapi._AsyncAdaptMilvusqlDBAPI
    )

"""Async DBAPI shim for the ``milvusql+aio`` driver (D12).

Wraps ``milvusql.aio`` (asyncio-native, built on
``pymilvus.AsyncMilvusClient``) using SQLAlchemy's own generic
async-adapter class,
``sqlalchemy.connectors.asyncio.AsyncAdapt_dbapi_connection`` -- no
custom connection/cursor subclass needed. ``milvusql.aio.AsyncConnection``/
``AsyncCursor`` already satisfy the ``AsyncIODBAPIConnection``/
``AsyncIODBAPICursor`` protocol that adapter expects (confirmed
directly against the protocol definitions in
``sqlalchemy.connectors.asyncio``: async ``close()``/``commit()``/
``rollback()`` on the connection, ``__aenter__``/``__aexit__`` and
async ``close()`` on the cursor -- all added to ``milvusql.aio``
specifically to satisfy this).

Unlike ``aiosqlite``'s dialect (the reference read while building this),
``milvusql.aio.connect()`` needs no ``await_only()`` dance at connect
time: it does no I/O itself (``AsyncMilvusClient()`` construction is
synchronous, same as the sync ``MilvusClient()``), so it's already a
ready, usable ``AsyncConnection`` the moment it returns.
"""

from __future__ import annotations

import typing as t

from sqlalchemy.connectors.asyncio import AsyncAdapt_dbapi_connection

import milvusql.aio
from milvusql.dbapi import errors

if t.TYPE_CHECKING:
    from sqlalchemy.connectors.asyncio import AsyncIODBAPIConnection


class _AsyncAdaptMilvusqlDBAPI:
    """Stands in for a DBAPI module -- ``import_dbapi()`` returns an
    instance of this, the same pattern ``aiosqlite``'s own dialect
    uses. Exposes ``connect()`` plus the exception classes SQLAlchemy
    expects to find on a DBAPI module, reusing ``milvusql.dbapi.errors``
    directly: the very same classes the sync dialect's error
    translation already produces, so ``isinstance`` checks against
    ``self.dbapi.Error``/``OperationalError``/... behave identically
    whether the engine is sync or async."""

    paramstyle = "named"

    Warning = errors.Warning
    Error = errors.Error
    InterfaceError = errors.InterfaceError
    DatabaseError = errors.DatabaseError
    DataError = errors.DataError
    OperationalError = errors.OperationalError
    IntegrityError = errors.IntegrityError
    InternalError = errors.InternalError
    ProgrammingError = errors.ProgrammingError
    NotSupportedError = errors.NotSupportedError

    def connect(
        self, *args: t.Any, **kwargs: t.Any
    ) -> AsyncAdapt_dbapi_connection:
        connection = milvusql.aio.connect(*args, **kwargs)
        # `AsyncIODBAPIConnection` declares `__getattr__`/`__setattr__`
        # (a catch-all escape hatch for DBAPIs whose real connection is a
        # thread/queue proxy, e.g. aiosqlite's) that a plain, concrete
        # class like `AsyncConnection` doesn't and shouldn't define --
        # it already has every attribute the protocol's structural check
        # (and the adapter's actual runtime usage) needs: async
        # `close`/`commit`/`rollback`, `cursor()`. Deliberate, understood
        # cast, same pattern as the sync dialect's `import_dbapi`.
        return AsyncAdapt_dbapi_connection(
            self, t.cast("AsyncIODBAPIConnection", connection)
        )


dbapi = _AsyncAdaptMilvusqlDBAPI()

__all__ = ["dbapi"]

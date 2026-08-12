"""PEP 249 module surface: ``apilevel``, ``threadsafety``,
``paramstyle``, ``connect()``, and the exception hierarchy every DBAPI
module must expose at module level."""

from __future__ import annotations

from milvusql.dbapi.connection import (
    Connection,
    connect,
    token_from_credentials,
)
from milvusql.dbapi.cursor import Cursor
from milvusql.dbapi.errors import (
    DatabaseError,
    DataError,
    Error,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    Warning,  # noqa: A004 -- PEP 249 mandates this name
)

apilevel = "2.0"
#: One thread may not share a cursor, but may open several connections
#: (D9: one ``MilvusClient``/gRPC channel per connection, no shared
#: mutable state across connections).
threadsafety = 1
#: MilvusQL binds are ``:name`` (Track A's README); PEP 249's name for
#: that spelling.
paramstyle = "named"

__all__ = [
    "Connection",
    "Cursor",
    "DataError",
    "DatabaseError",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Warning",
    "apilevel",
    "connect",
    "paramstyle",
    "threadsafety",
    "token_from_credentials",
]

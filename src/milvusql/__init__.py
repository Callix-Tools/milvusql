"""milvusql -- a PEP 249 DBAPI (sync + async) for Milvus.

Parses/generates MilvusQL via ``sqlglot-milvus`` and executes the
resulting AST against ``pymilvus``. See ``milvusql.aio`` for the
asyncio-native client built on the same dispatch table.
"""

from __future__ import annotations

from milvusql.dbapi import connect
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

__version__ = "0.1.0"

__all__ = [
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
    "__version__",
    "connect",
]

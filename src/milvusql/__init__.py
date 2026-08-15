"""milvusql -- a PEP 249 DBAPI (sync + async) for Milvus.

Parses/generates MilvusQL via ``sqlglot-milvus`` and executes the
resulting AST against ``pymilvus``. See ``milvusql.aio`` for the
asyncio-native client built on the same dispatch table.
"""

from __future__ import annotations

import importlib.metadata as _metadata

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

# Single-sourced from the installed package's metadata (pyproject's
# `version`, which the release pipeline bumps) -- a hardcoded string
# here silently drifted from the released version until it was read
# from the one place `uv version` actually writes.
try:
    __version__ = _metadata.version("milvusql")
except _metadata.PackageNotFoundError:  # pragma: no cover -- source tree only
    __version__ = "0.0.0"

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
